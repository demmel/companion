"""
FastAPI server for single-user agent system
"""

from agent.logging_config import setup_logging

setup_logging()

import logging
from typing import Literal
import uuid
import shutil
from pathlib import Path

from agent.api_types.api import (
    AutoWakeupSetRequest,
    AutoWakeupSetResponse,
    AutoWakeupStatusResponse,
    ImageUploadResponse,
    ModelConfigResponse,
    ModelConfigUpdateRequest,
    ModelConfigUpdateResponse,
    ResetResponse,
)
from agent.llm import create_llm, SupportedModel
from typing import List, Optional, Union
from datetime import datetime

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent.core import Agent
from agent.paths import agent_paths
from agent.api_types.events import AgentErrorEvent, EventEnvelope
from agent.agent_event_manager import AgentEventManager
from agent.api_types.timeline import (
    TimelineResponse,
)
from agent.config import Config
from agent.llm.models import ModelConfig
from pydantic import BaseModel, TypeAdapter, ValidationError


logger = logging.getLogger(__name__)


def initialize_agent(load: bool) -> AgentEventManager:
    """Initialize the agent with specific conversation files for development"""
    llm = create_llm()

    # Create manager first (it will be the event emitter)
    manager = AgentEventManager(agent=None)  # type: ignore - will set agent next

    # Create agent with manager as event emitter
    agent = Agent(
        llm=llm,
        model_config=Config.get_model_config(),
        event_emitter=manager,
        enable_image_generation=True,
        auto_summarize_threshold=16000,
        individual_trigger_compression=True,
        auto_save=True,
    )

    # Set the agent in the manager
    manager.agent = agent

    if load:
        try:
            agent.load_conversation("baseline")
        except Exception as e:
            import traceback

            logger.error(f"Failed to load conversation: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.info("Starting with a fresh agent instead")

    return manager


app = FastAPI(
    title="Agent API",
    description="Single-User Streaming AI Agent API",
    version="1.0.0",
)

app.state.agent_manager = initialize_agent(
    load=True  # Set to True to load specific conversation for development
)

# Add CORS middleware for local network access (phone, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from any device on local network
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/context")
async def get_context_info():
    """Get current context information"""
    manager: AgentEventManager = app.state.agent_manager
    context_info = manager.get_context_info()
    return {
        "message_count": context_info.message_count,
        "conversation_messages": context_info.conversation_messages,
        "estimated_tokens": context_info.estimated_tokens,
        "context_limit": context_info.context_limit,
        "usage_percentage": context_info.usage_percentage,
        "approaching_limit": context_info.approaching_limit,
    }


@app.get("/api/timeline", response_model=TimelineResponse)
async def get_timeline(
    page_size: int = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    """Get paginated timeline in chronological order, defaulting to most recent page"""
    from agent.api_types.timeline import build_timeline_page

    manager: AgentEventManager = app.state.agent_manager
    trigger_history = manager.get_trigger_history()
    all_entries = trigger_history.get_all_entries()

    # Parse cursor indices
    before_index = None
    after_index = None

    if before is not None:
        try:
            before_index = int(before)
        except (ValueError, TypeError):
            logger.warning(f"Invalid before cursor: {before}")

    if after is not None:
        try:
            after_index = int(after)
        except (ValueError, TypeError):
            logger.warning(f"Invalid after cursor: {after}")

    # Build timeline page using shared utility
    page_entries, pagination = build_timeline_page(
        all_entries, page_size, before_index, after_index
    )

    logger.info(
        f"Timeline request: page_size={page_size}, after={after}, before={before}, "
        f"returned {len(page_entries)} entries, total={pagination.total_items}"
    )

    return TimelineResponse(
        entries=page_entries,
        pagination=pagination,
    )


@app.post("/api/reset", response_model=ResetResponse)
async def reset_agent():
    """Reset the agent"""

    # Get the old manager to transfer state and clean up resources
    old_manager: AgentEventManager | None = app.state.agent_manager
    current_client_queue = None

    if old_manager:
        # Disable wakeup timer scheduling and cancel any active timer
        old_manager.set_auto_wakeup_enabled(False)

        # Disable auto-save to prevent the old agent from saving mid-reset
        old_manager.auto_save = False

        # Get the current client queue to transfer to new manager
        with old_manager.client_queue_lock:
            current_client_queue = old_manager.current_client_queue
            # Clear the old manager's queue reference so it stops pushing events
            old_manager.current_client_queue = None

    # Reinitialize the agent manager
    new_manager = initialize_agent(
        load=False  # Set to False to avoid loading specific conversation
    )

    # Transfer the client queue to the new manager if one exists
    if current_client_queue is not None:
        # Clear any remaining events from the old agent before transferring
        while not current_client_queue.empty():
            try:
                current_client_queue.get_nowait()
            except:
                break

        new_manager.set_client_queue(current_client_queue)

    app.state.agent_manager = new_manager

    return ResetResponse(
        message="Agent reset successfully",
        timestamp=datetime.now().isoformat(),
    )


class ClientSendMessageRequest(BaseModel):
    """Message received from client over WebSocket"""

    type: Literal["message"] = "message"
    message: str
    user_name: str
    image_ids: Optional[List[str]] = None


class ClientHydrationRequest(BaseModel):
    """Hydration request from client over WebSocket"""

    type: Literal["hydrate"] = "hydrate"
    last_trigger_id: Optional[str] = None
    last_event_sequence: Optional[int] = None


ClientRequest = Union[ClientSendMessageRequest, ClientHydrationRequest]
ClientRequestAdapter: TypeAdapter[ClientRequest] = TypeAdapter(ClientRequest)


@app.websocket("/api/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()

    import asyncio
    import threading
    import queue as queue_module

    # Create client-specific queue and register with manager (replaces any existing client)
    manager: AgentEventManager = app.state.agent_manager
    client_queue: queue_module.Queue[EventEnvelope] = queue_module.Queue()
    manager.set_client_queue(client_queue)

    logger.info("WebSocket client connected, queue registered")

    try:

        async def handle_incoming_messages():
            """Handle incoming messages from client"""
            from typing import assert_never

            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    client_request = ClientRequestAdapter.validate_json(data)

                    match client_request:
                        case ClientHydrationRequest():
                            # Handle hydration request
                            logger.info(
                                f"Hydration request: trigger_id={client_request.last_trigger_id}, sequence={client_request.last_event_sequence}"
                            )

                            # Get hydration events (returns List[AgentServerEvent])
                            server_events = manager.get_hydration_events(
                                last_trigger_id=client_request.last_trigger_id,
                                last_event_sequence=client_request.last_event_sequence,
                            )

                            logger.info(
                                f"Sending {len(server_events)} hydration events"
                            )
                            for i, server_event in enumerate(server_events):
                                logger.debug(
                                    f"Sending event {i+1}/{len(server_events)}: {server_event.type}"
                                )
                                await websocket.send_text(
                                    server_event.model_dump_json()
                                )
                            logger.info(
                                f"Finished sending {len(server_events)} hydration events"
                            )

                        case ClientSendMessageRequest():
                            # Handle sending message to agent
                            message = client_request.message
                            image_ids = client_request.image_ids or []
                            user_name = client_request.user_name

                            # Resolve image IDs to file paths
                            image_paths = None
                            if image_ids:
                                upload_dir = agent_paths.get_uploaded_images_dir()
                                image_paths = []
                                for image_id in image_ids:
                                    # Find the image file with this ID (could be any supported extension)
                                    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                                        image_file = upload_dir / f"{image_id}{ext}"
                                        if image_file.exists():
                                            image_paths.append(str(image_file))
                                            break
                                    else:
                                        logger.warning(
                                            f"Image not found for ID: {image_id}"
                                        )

                            # Create appropriate trigger
                            if not message.strip() and not image_paths:
                                from agent.chain_of_action.trigger import WakeupTrigger

                                trigger = WakeupTrigger()
                            else:
                                from agent.chain_of_action.trigger import (
                                    UserInputTrigger,
                                )

                                trigger = UserInputTrigger(
                                    content=message,
                                    user_name=user_name,
                                    image_paths=image_paths,
                                )

                            # Process message in background thread
                            logger.info(
                                f"Processing trigger: {trigger.model_dump_json(indent=2)}"
                            )

                            def process_message():
                                try:
                                    manager.chat_stream(trigger=trigger)
                                except Exception as e:
                                    # Put error event in queue
                                    error_event = AgentErrorEvent(
                                        message=f"Internal error: {str(e)}"
                                    )
                                    manager.emit(error_event)

                            # Run agent processing in background thread
                            thread = threading.Thread(target=process_message)
                            thread.start()

                        case _:
                            assert_never(client_request)

            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")

        async def handle_outgoing_events():
            """Handle outgoing events to client"""
            try:
                while True:
                    # Get envelope from our local client queue with timeout
                    try:
                        envelope = await asyncio.to_thread(client_queue.get, True, 1.0)
                        await websocket.send_text(envelope.model_dump_json())
                    except queue_module.Empty:
                        # Timeout - check if WebSocket is still alive
                        if (
                            websocket.client_state
                            == websocket.client_state.DISCONNECTED
                        ):
                            logger.info(
                                "WebSocket disconnected during timeout, stopping event handler"
                            )
                            break
                        continue
                    except WebSocketDisconnect:
                        logger.info("WebSocket disconnect detected in outgoing events")
                        break
                    except Exception as e:
                        # Log the error but check if it's due to closed connection
                        if "websocket.close" in str(
                            e
                        ) or "response already completed" in str(e):
                            logger.info("WebSocket closed, stopping event handler")
                            break
                        else:
                            logger.error(f"Queue/WebSocket error: {e}")
                            break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnect in event handler")
            except Exception as e:
                logger.error(f"WebSocket event sending error: {e}")

        # Run both handlers concurrently
        await asyncio.gather(handle_incoming_messages(), handle_outgoing_events())
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clear our queue from the manager (only if it's still ours)
        manager.clear_client_queue(client_queue)
        logger.info("WebSocket client disconnected, queue cleared")


@app.get("/api/auto-wakeup", response_model=AutoWakeupStatusResponse)
async def get_auto_wakeup_status():
    """Get current auto-wakeup status"""
    manager: AgentEventManager = app.state.agent_manager
    return AutoWakeupStatusResponse(
        enabled=manager.get_auto_wakeup_enabled(),
        delay_seconds=manager.wakeup_delay_seconds,
    )


@app.post("/api/auto-wakeup", response_model=AutoWakeupSetResponse)
async def set_auto_wakeup_status(request: AutoWakeupSetRequest):
    """Set auto-wakeup enabled state"""
    manager: AgentEventManager = app.state.agent_manager
    manager.set_auto_wakeup_enabled(request.enabled)

    return AutoWakeupSetResponse(
        enabled=manager.get_auto_wakeup_enabled(),
        message=f"Auto-wakeup {'enabled' if request.enabled else 'disabled'}",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/model-config", response_model=ModelConfigResponse)
async def get_model_config():
    """Get current model configuration for all action types"""
    model_config = Config.get_model_config()

    return ModelConfigResponse(
        state_initialization_model=model_config.state_initialization_model.value,
        action_planning_model=model_config.action_planning_model.value,
        situational_analysis_model=model_config.situational_analysis_model.value,
        memory_retrieval_model=model_config.memory_retrieval_model.value,
        memory_formation_model=model_config.memory_formation_model.value,
        think_action_model=model_config.think_action_model.value,
        speak_action_model=model_config.speak_action_model.value,
        visual_action_model=model_config.visual_action_model.value,
        fetch_url_action_model=model_config.fetch_url_action_model.value,
        evaluate_priorities_action_model=model_config.evaluate_priorities_action_model.value,
    )


@app.post("/api/model-config", response_model=ModelConfigUpdateResponse)
async def update_model_config(request: ModelConfigUpdateRequest):
    """Update model configuration for all action types"""
    try:
        # Validate that all provided models are valid SupportedModel values
        new_config = ModelConfig(
            state_initialization_model=SupportedModel(
                request.state_initialization_model
            ),
            action_planning_model=SupportedModel(request.action_planning_model),
            situational_analysis_model=SupportedModel(
                request.situational_analysis_model
            ),
            memory_retrieval_model=SupportedModel(request.memory_retrieval_model),
            memory_formation_model=SupportedModel(request.memory_formation_model),
            think_action_model=SupportedModel(request.think_action_model),
            speak_action_model=SupportedModel(request.speak_action_model),
            visual_action_model=SupportedModel(request.visual_action_model),
            fetch_url_action_model=SupportedModel(request.fetch_url_action_model),
            evaluate_priorities_action_model=SupportedModel(
                request.evaluate_priorities_action_model
            ),
        )

        # Save the configuration
        Config.set_model_config(new_config)
        agent: Agent = app.state.agent_manager.agent
        if agent:
            agent.model_config = new_config

        # Return the updated configuration
        return ModelConfigUpdateResponse(
            message="Model configuration updated successfully",
            timestamp=datetime.now().isoformat(),
            config=ModelConfigResponse(
                state_initialization_model=new_config.state_initialization_model.value,
                action_planning_model=new_config.action_planning_model.value,
                situational_analysis_model=new_config.situational_analysis_model.value,
                memory_retrieval_model=new_config.memory_retrieval_model.value,
                memory_formation_model=new_config.memory_formation_model.value,
                think_action_model=new_config.think_action_model.value,
                speak_action_model=new_config.speak_action_model.value,
                visual_action_model=new_config.visual_action_model.value,
                fetch_url_action_model=new_config.fetch_url_action_model.value,
                evaluate_priorities_action_model=new_config.evaluate_priorities_action_model.value,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {str(e)}")


@app.post("/api/upload-image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file and return a unique ID"""

    # Check file size before processing (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}",
        )

    # Generate unique ID and filename
    image_id = str(uuid.uuid4())
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_extension = Path(file.filename).suffix
    if not file_extension:
        raise HTTPException(status_code=400, detail="File must have a valid extension")

    new_filename = f"{image_id}{file_extension}"

    # Get upload directory and ensure it exists
    upload_dir = agent_paths.get_uploaded_images_dir()
    upload_dir.mkdir(exist_ok=True)

    # Save file
    file_path = upload_dir / new_filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Failed to save image")

    # Get file size for response
    file_size = file_path.stat().st_size

    logger.info(f"Image uploaded: {image_id} ({file.filename}, {file_size} bytes)")

    return ImageUploadResponse(
        id=image_id,
        size=file_size,
        url=f"/uploaded_images/{new_filename}",
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""

    logger.info("Health check requested")

    manager: AgentEventManager = app.state.agent_manager
    return {
        "status": "healthy",
        "agent_initialized": manager is not None,
        "agent_name": manager.state.name if manager.state else None,
        "timestamp": datetime.now().isoformat(),
    }


# Static files configuration using centralized paths
client_dist_dir = agent_paths.get_client_dist_dir()

if client_dist_dir.exists():
    logger.info(f"✅ Serving React client from: {client_dist_dir}")

    # Mount static assets (JS, CSS, images, etc.)
    app.mount(
        "/assets",
        StaticFiles(directory=agent_paths.get_client_assets_dir()),
        name="assets",
    )

    # Mount generated images directory
    app.mount(
        "/generated_images",
        StaticFiles(directory=agent_paths.get_generated_images_dir()),
        name="generated_images",
    )

    # Mount uploaded images directory
    app.mount(
        "/uploaded_images",
        StaticFiles(directory=agent_paths.get_uploaded_images_dir()),
        name="uploaded_images",
    )

    # Catch-all route for React SPA (must be last!)
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve React SPA, fallback to index.html for client-side routing"""

        # Try to serve specific file first
        file_path = client_dist_dir / path
        if file_path.is_file():
            return FileResponse(file_path)

        # Fallback to index.html for SPA routing
        index_html_path = agent_paths.get_client_index_html()
        if index_html_path.exists():
            return FileResponse(index_html_path)

        return {"message": "React client not built. Run 'cd client && npm run build'"}

else:
    logger.warning(f"⚠️  React client not found at: {client_dist_dir}")
    logger.warning("   Run 'cd client && npm run build' to build the client first")

    # Provide helpful message at root
    @app.get("/")
    async def no_client():
        return {
            "message": "Agent API Server",
            "client_status": "not_built",
            "instructions": "Run 'cd client && npm run build' to enable web interface",
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Please use uvicorn to run the server:")
    print()
    print("  Development:")
    print("    uvicorn agent.api_server:app --reload")
    print()
    print("  Production:")
    print("    uvicorn agent.api_server:app --host 0.0.0.0 --port 8000")
    print("=" * 60)
