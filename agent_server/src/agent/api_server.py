"""
FastAPI server for single-user agent system
"""

from contextlib import asynccontextmanager
from collections.abc import Callable as AbcCallable
from agent.logging_config import setup_logging

setup_logging()

import logging
from typing import Annotated, Callable, Literal
import uuid
import shutil
from pathlib import Path

from agent.api_types.api import (
    AutoWakeupSetRequest,
    AutoWakeupSetResponse,
    AutoWakeupStatusResponse,
    ImageUploadResponse,
    InstalledOllamaModelResponse,
    InstalledOllamaModelsResponse,
    ModelConfigResponse,
    ModelConfigUpdateRequest,
    ModelConfigUpdateResponse,
    OllamaModelMutationResponse,
    PullOllamaModelRequest,
    ResetResponse,
    SupportedModelsResponse,
)
from agent.llm import create_llm, SupportedModel
from typing import List, Optional, Union
from datetime import datetime
import ollama
from ollama import _types as ollama_types

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent.tts import RenderStatus

from agent.core import Agent
from agent.paths import agent_paths
from agent.api_types.events import AgentErrorEvent, EventEnvelope
from agent.agent_event_manager import AgentEventManager
from agent.api_types.timeline import (
    TimelineResponse,
)
from agent.config import Config
from agent.llm.models import (
    KNOWN_ANTHROPIC_MODELS,
    KNOWN_OLLAMA_MODELS,
    ModelConfig,
    is_anthropic_model,
)
from pydantic import BaseModel, TypeAdapter, ValidationError


logger = logging.getLogger(__name__)


def initialize_agent(load: bool) -> AgentEventManager:
    """Initialize the agent with specific conversation files for development"""
    llm = create_llm()

    # Create manager first (it will be the event emitter)
    manager = AgentEventManager(agent=None)  # type: ignore - will set agent next

    # Create agent using factory methods
    if load:
        try:
            agent = Agent.load(
                conversation_id="baseline",
                llm=llm,
                model_config=Config.get_model_config(),
                event_emitter=manager,
                enable_image_generation=True,
                auto_summarize_threshold=32768,
                individual_trigger_compression=False,
            )
        except Exception as e:
            import traceback

            logger.error(f"Failed to load conversation: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.info("Starting with a fresh agent instead")
            agent = Agent.new(
                llm=llm,
                model_config=Config.get_model_config(),
                event_emitter=manager,
                enable_image_generation=True,
                auto_summarize_threshold=32768,
                individual_trigger_compression=False,
            )
    else:
        agent = Agent.new(
            llm=llm,
            model_config=Config.get_model_config(),
            event_emitter=manager,
            enable_image_generation=True,
            auto_summarize_threshold=32768,
            individual_trigger_compression=False,
        )

    # Set the agent in the manager
    manager.agent = agent

    return manager


def _set_agent_manager(app: FastAPI, manager: AgentEventManager | None) -> None:
    app.state.agent_manager = manager
    app.state.agent_startup_error = None


def _record_agent_startup_error(app: FastAPI, exc: Exception) -> None:
    logger.error("Agent startup failed: %s", exc, exc_info=True)
    app.state.agent_manager = None
    app.state.agent_startup_error = str(exc)


def _get_agent_manager(app: FastAPI) -> AgentEventManager:
    manager = app.state.agent_manager
    if manager is None:
        startup_error = app.state.agent_startup_error
        detail = "Agent is unavailable"
        if startup_error:
            detail = f"{detail}: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)
    return manager


def get_manager(request: Request) -> AgentEventManager:
    return _get_agent_manager(request.app)


ManagerDep = Annotated[AgentEventManager, Depends(get_manager)]


def _configure_static_routes(app: FastAPI) -> None:
    # Static files configuration using centralized paths
    client_dist_dir = agent_paths.get_client_dist_dir()

    if client_dist_dir.exists():
        logger.info(f"Serving React client from: {client_dist_dir}")

        app.mount(
            "/assets",
            StaticFiles(directory=agent_paths.get_client_assets_dir()),
            name="assets",
        )
        app.mount(
            "/generated_images",
            StaticFiles(directory=agent_paths.get_generated_images_dir()),
            name="generated_images",
        )
        app.mount(
            "/uploaded_images",
            StaticFiles(directory=agent_paths.get_uploaded_images_dir()),
            name="uploaded_images",
        )
        app.mount(
            "/generated_audio",
            StaticFiles(directory=agent_paths.get_generated_audio_dir()),
            name="generated_audio",
        )

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            """Serve React SPA, fallback to index.html for client-side routing"""

            file_path = client_dist_dir / path
            if file_path.is_file():
                return FileResponse(file_path)

            index_html_path = agent_paths.get_client_index_html()
            if index_html_path.exists():
                return FileResponse(index_html_path)

            return {"message": "React client not built. Run 'cd client && npm run build'"}

        return

    logger.warning(f"React client not found at: {client_dist_dir}")
    logger.warning("Run 'cd client && npm run build' to build the client first")

    @app.get("/")
    async def no_client():
        return {
            "message": "Agent API Server",
            "client_status": "not_built",
            "instructions": "Run 'cd client && npm run build' to enable web interface",
        }


def _set_ollama_client_factory(
    app: FastAPI, factory: AbcCallable[[], ollama.Client]
) -> None:
    app.state.ollama_client_factory = factory


def _get_ollama_client(app: FastAPI) -> ollama.Client:
    factory = app.state.ollama_client_factory
    if factory is None:
        raise HTTPException(status_code=500, detail="Ollama client is unavailable")
    return factory()


def _normalize_ollama_model(
    model: ollama_types.ListResponse.Model,
) -> InstalledOllamaModelResponse:
    return InstalledOllamaModelResponse(
        name=model.model,
        size=model.size,
        modified_at=model.modified_at.isoformat() if model.modified_at else None,
        digest=model.digest,
        details=model.details.model_dump(exclude_none=True) if model.details else {},
    )


def _map_ollama_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ollama.ResponseError):
        status_code = exc.status_code if exc.status_code > 0 else 502
        return HTTPException(status_code=status_code, detail=exc.error)
    return HTTPException(status_code=502, detail=f"Failed to reach Ollama: {exc}")


def _get_model_config_references(model_name: str) -> list[str]:
    configured_fields: list[str] = []
    for field_name, value in Config.get_model_config().__dict__.items():
        if str(value) == model_name:
            configured_fields.append(field_name)
    return configured_fields


def _serialize_model_config(model_config: ModelConfig) -> ModelConfigResponse:
    return ModelConfigResponse(
        state_initialization_model=str(model_config.state_initialization_model),
        action_planning_model=str(model_config.action_planning_model),
        situational_analysis_model=str(model_config.situational_analysis_model),
        memory_retrieval_model=str(model_config.memory_retrieval_model),
        memory_formation_model=str(model_config.memory_formation_model),
        trigger_compression_model=str(model_config.trigger_compression_model),
        think_action_model=str(model_config.think_action_model),
        speak_action_model=str(model_config.speak_action_model),
        visual_action_model=str(model_config.visual_action_model),
        fetch_url_action_model=str(model_config.fetch_url_action_model),
        evaluate_priorities_action_model=str(model_config.evaluate_priorities_action_model),
        tts_rewrite_model=str(model_config.tts_rewrite_model),
    )


def _validate_model_name(model_name: str) -> SupportedModel:
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("model name cannot be empty")
    model = SupportedModel(model_name)
    if is_anthropic_model(model):
        return model
    if model_name.startswith("claude-"):
        raise ValueError(
            f"{model_name!r} is not a known Anthropic model; use one of the supported Claude model IDs"
        )
    return model


def create_app(
    *,
    agent_manager_factory: Callable[[], AgentEventManager] = lambda: initialize_agent(
        load=True
    ),
    initialize_on_startup: bool = True,
    ollama_client_factory: AbcCallable[[], ollama.Client] = lambda: ollama.Client(
        host=Config.ollama_host()
    ),
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _set_agent_manager(app, None)

        if initialize_on_startup:
            try:
                _set_agent_manager(app, agent_manager_factory())
            except Exception as exc:
                _record_agent_startup_error(app, exc)

        yield

        if app.state.agent_manager is not None:
            app.state.agent_manager.agent.close()

    app = FastAPI(
        title="Agent API",
        description="Single-User Streaming AI Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Initialise state attributes eagerly so they can always be accessed directly
    app.state.agent_manager = None
    app.state.agent_startup_error = None
    _set_ollama_client_factory(app, ollama_client_factory)
    return app


app = create_app()

# Add CORS middleware for local network access (phone, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from any device on local network
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/context")
async def get_context_info(manager: ManagerDep):
    """Get current context information"""
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
    manager: ManagerDep,
    page_size: int = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    """Get paginated timeline in chronological order, defaulting to most recent page"""
    from agent.api_types.timeline import build_timeline_page

    trigger_history = manager.get_trigger_history()

    # Parse cursor indices
    before_index: Optional[int] = None
    after_index: Optional[int] = None

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
        trigger_history, page_size, before_index, after_index
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
    old_manager: AgentEventManager | None = getattr(app.state, "agent_manager", None)
    current_client_queue = None

    if old_manager:
        # Disable wakeup timer scheduling and cancel any active timer
        old_manager.set_auto_wakeup_enabled(False)

        # Get the current client queue to transfer to new manager
        with old_manager.client_queue_lock:
            current_client_queue = old_manager.current_client_queue
            # Clear the old manager's queue reference so it stops pushing events
            old_manager.current_client_queue = None

        # Reclaim the old agent's resources (LLM worker thread, TTS, SQLite /
        # ChromaDB handles). Done off the event loop once any in-flight run has
        # finished so llm.close()'s blocking join doesn't stall the server and
        # we don't tear down the trigger history under a running reasoning loop.
        old_manager.close_when_idle()

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

    _set_agent_manager(app, new_manager)

    return ResetResponse(
        message="Agent reset successfully",
        timestamp=datetime.now().isoformat(),
    )


class RegenerateImageRequest(BaseModel):
    """Request to regenerate an image from existing metadata"""

    trigger_id: str
    action_index: int


class RegenerateImageResponse(BaseModel):
    """Response for image regeneration request"""

    success: bool
    new_image_url: Optional[str] = None
    error: Optional[str] = None


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
    manager = _get_agent_manager(app)
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

                    # Resolve the current manager fresh for every request so that
                    # a reset (which swaps the manager in app.state) routes new
                    # work to the live manager instead of the stale one captured
                    # when this socket connected.
                    current_manager = _get_agent_manager(app)

                    match client_request:
                        case ClientHydrationRequest():
                            # Handle hydration request
                            logger.info(
                                f"Hydration request: trigger_id={client_request.last_trigger_id}, sequence={client_request.last_event_sequence}"
                            )

                            # Get hydration events (returns List[AgentServerEvent])
                            server_events = current_manager.get_hydration_events(
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
                                # Resolve the manager again inside the thread so a
                                # reset racing the thread start still routes to the
                                # live manager (and emits to the attached queue).
                                msg_manager = _get_agent_manager(app)
                                try:
                                    msg_manager.chat_stream(trigger=trigger)
                                except Exception as e:
                                    # Put error event in queue
                                    error_event = AgentErrorEvent(
                                        message=f"Internal error: {str(e)}"
                                    )
                                    msg_manager.emit(error_event)

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
        # Clear our queue from whichever manager currently holds it. After a
        # reset the queue is transferred to the new manager, so clearing the
        # manager captured at connect would leak the queue (the new manager
        # would keep emitting into this dead client's queue).
        cleanup_manager = app.state.agent_manager or manager
        cleanup_manager.clear_client_queue(client_queue)
        logger.info("WebSocket client disconnected, queue cleared")


@app.get("/api/auto-wakeup", response_model=AutoWakeupStatusResponse)
async def get_auto_wakeup_status(manager: ManagerDep):
    """Get current auto-wakeup status"""
    return AutoWakeupStatusResponse(
        enabled=manager.get_auto_wakeup_enabled(),
        delay_seconds=manager.wakeup_delay_seconds,
    )


@app.post("/api/auto-wakeup", response_model=AutoWakeupSetResponse)
async def set_auto_wakeup_status(request: AutoWakeupSetRequest, manager: ManagerDep):
    """Set auto-wakeup enabled state"""
    manager.set_auto_wakeup_enabled(request.enabled)

    return AutoWakeupSetResponse(
        enabled=manager.get_auto_wakeup_enabled(),
        message=f"Auto-wakeup {'enabled' if request.enabled else 'disabled'}",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/supported-models", response_model=SupportedModelsResponse)
async def get_supported_models(request: Request):
    """Get list of supported models: all known Anthropic models plus currently installed Ollama models."""
    try:
        response = _get_ollama_client(request.app).list()
        ollama_models = sorted(
            [m.model for m in response.models if m.model],
            key=str.lower,
        )
    except Exception:
        # Ollama unavailable – fall back to known model suggestions
        ollama_models = [str(m) for m in KNOWN_OLLAMA_MODELS]

    anthropic_models = [str(m) for m in KNOWN_ANTHROPIC_MODELS]
    return SupportedModelsResponse(
        models=[*anthropic_models, *ollama_models],
        ollama_models=ollama_models,
    )


@app.get("/api/ollama/models", response_model=InstalledOllamaModelsResponse)
async def get_installed_ollama_models(request: Request):
    """Get the list of models currently installed in Ollama."""
    try:
        response = _get_ollama_client(request.app).list()
    except Exception as exc:
        raise _map_ollama_error(exc)

    models = response.models
    normalized_models = sorted(
        (_normalize_ollama_model(model) for model in models),
        key=lambda model: model.name.lower(),
    )
    return InstalledOllamaModelsResponse(models=normalized_models)


@app.post(
    "/api/ollama/models/pull",
    response_model=OllamaModelMutationResponse,
)
async def pull_ollama_model(request: PullOllamaModelRequest, http_request: Request):
    """Pull an Ollama model by name."""
    try:
        _get_ollama_client(http_request.app).pull(request.name)
    except Exception as exc:
        raise _map_ollama_error(exc)

    return OllamaModelMutationResponse(
        name=request.name,
        message=f"Pulled Ollama model '{request.name}'",
        timestamp=datetime.now().isoformat(),
    )


@app.delete(
    "/api/ollama/models/{model_name:path}",
    response_model=OllamaModelMutationResponse,
)
async def delete_ollama_model(model_name: str, request: Request):
    """Delete an installed Ollama model by name."""
    configured_fields = _get_model_config_references(model_name)
    if configured_fields:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot delete Ollama model '{model_name}' because it is still "
                f"configured for: {', '.join(configured_fields)}"
            ),
        )

    try:
        _get_ollama_client(request.app).delete(model_name)
    except Exception as exc:
        raise _map_ollama_error(exc)

    return OllamaModelMutationResponse(
        name=model_name,
        message=f"Deleted Ollama model '{model_name}'",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/model-config", response_model=ModelConfigResponse)
async def get_model_config():
    """Get current model configuration for all action types"""
    model_config = Config.get_model_config()

    return _serialize_model_config(model_config)


@app.post("/api/model-config", response_model=ModelConfigUpdateResponse)
async def update_model_config(request: ModelConfigUpdateRequest, manager: ManagerDep):
    """Update model configuration for all action types"""
    try:
        new_config = ModelConfig(
            state_initialization_model=_validate_model_name(request.state_initialization_model),
            action_planning_model=_validate_model_name(request.action_planning_model),
            situational_analysis_model=_validate_model_name(request.situational_analysis_model),
            memory_retrieval_model=_validate_model_name(request.memory_retrieval_model),
            memory_formation_model=_validate_model_name(request.memory_formation_model),
            trigger_compression_model=_validate_model_name(request.trigger_compression_model),
            think_action_model=_validate_model_name(request.think_action_model),
            speak_action_model=_validate_model_name(request.speak_action_model),
            visual_action_model=_validate_model_name(request.visual_action_model),
            fetch_url_action_model=_validate_model_name(request.fetch_url_action_model),
            evaluate_priorities_action_model=_validate_model_name(
                request.evaluate_priorities_action_model
            ),
            tts_rewrite_model=_validate_model_name(request.tts_rewrite_model),
        )

        # Save the configuration
        Config.set_model_config(new_config)
        agent: Agent = manager.agent
        if agent:
            agent.model_config = new_config

        # Return the updated configuration
        return ModelConfigUpdateResponse(
            message="Model configuration updated successfully",
            timestamp=datetime.now().isoformat(),
            config=_serialize_model_config(new_config),
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


@app.post("/api/regenerate-image", response_model=RegenerateImageResponse)
async def regenerate_image(request: RegenerateImageRequest, manager: ManagerDep):
    """Regenerate an image using existing prompt with new random seed"""

    trigger_history = manager.get_trigger_history()

    # Find the trigger entry
    try:
        entry = trigger_history.get_entry_by_id(request.trigger_id)
    except Exception as e:
        logger.error(f"Failed to find trigger: {e}")
        return RegenerateImageResponse(
            success=False, error=f"Trigger not found: {request.trigger_id}"
        )

    # Get the action by index
    if request.action_index < 0 or request.action_index >= len(entry.actions_taken):
        return RegenerateImageResponse(
            success=False, error=f"Invalid action index: {request.action_index}"
        )

    action = entry.actions_taken[request.action_index]

    # Extract ImageGenerationToolContent from action result
    from agent.chain_of_action.action.base_action_data import ActionSuccessResult
    from agent.types import ImageGenerationToolContent
    from agent.chain_of_action.action.actions.visual_actions import (
        UpdateAppearanceOutput,
        UpdateEnvironmentOutput,
    )

    if action.result.type != "success":
        return RegenerateImageResponse(
            success=False, error="Action did not complete successfully"
        )

    # Get the image result from the action output
    result_content = action.result.content
    image_result = None

    if isinstance(result_content, (UpdateAppearanceOutput, UpdateEnvironmentOutput)):
        image_result = result_content.image_result

    if image_result is None or not isinstance(image_result, ImageGenerationToolContent):
        return RegenerateImageResponse(
            success=False, error="Action does not contain image generation metadata"
        )

    # Get image generation service
    from agent.image_generation import get_shared_image_generator

    image_service = get_shared_image_generator()

    # Progress callback (we don't stream progress for regeneration yet)
    def progress_callback(progress_data):
        logger.debug(f"Regeneration progress: {progress_data}")

    # Regenerate the image
    try:
        new_image_content = image_service.regenerate_from_metadata(
            metadata=image_result,
            progress_callback=progress_callback,
        )

        if new_image_content is None:
            return RegenerateImageResponse(
                success=False, error="Image generation failed"
            )

        # Update the action's result with new image
        if isinstance(result_content, UpdateAppearanceOutput):
            result_content.image_result = new_image_content
            result_content.image_description = new_image_content.prompt
        elif isinstance(result_content, UpdateEnvironmentOutput):
            result_content.image_result = new_image_content
            result_content.image_description = new_image_content.prompt

        # Save conversation to persist changes
        manager.agent.save_conversation()

        logger.info(
            f"Successfully regenerated image for action at index {request.action_index}"
        )

        return RegenerateImageResponse(
            success=True, new_image_url=new_image_content.image_url
        )

    except Exception as e:
        logger.error(f"Failed to regenerate image: {e}")
        import traceback

        traceback.print_exc()
        return RegenerateImageResponse(
            success=False, error=f"Image generation error: {str(e)}"
        )


@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint"""

    logger.info("Health check requested")

    manager = getattr(request.app.state, "agent_manager", None)
    startup_error = getattr(request.app.state, "agent_startup_error", None)
    return {
        "status": "healthy" if manager is not None else "degraded",
        "agent_initialized": manager is not None,
        "agent_name": manager.state.name if manager and manager.state else None,
        "startup_error": startup_error,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def root_health_check(request: Request):
    """Root health check for deployment probes."""
    return await health_check(request)


def _try_queue_tts_render(agent: Agent, trigger_id: str, action_index: int) -> bool:
    """Try to queue TTS render for a historical action. Returns True if queued."""
    from agent.chain_of_action.action.actions.speak_action import SpeakActionData
    from agent.chain_of_action.action.actions.think_action import ThinkActionData

    if agent.tts_service is None:
        return False

    try:
        trigger_history = agent.get_trigger_history()
        entry = trigger_history.get_entry_by_id(trigger_id)

        if action_index < 0 or action_index >= len(entry.actions_taken):
            return False

        action = entry.actions_taken[action_index]

        if action.result.type != "success":
            return False

        action_id = f"{trigger_id}_{action_index}"

        match action:
            case SpeakActionData():
                text = action.result.content.response
                tone = action.input.tone
                agent.tts_service.queue_render(action_id, text, tone)
                logger.info(f"Queued on-demand TTS render for speak action {action_id}")
                return True
            case ThinkActionData():
                text = action.result.content.thoughts
                agent.tts_service.queue_render(action_id, text, None)
                logger.info(f"Queued on-demand TTS render for think action {action_id}")
                return True

        return False
    except Exception as e:
        logger.error(f"Failed to queue TTS render: {e}")
        return False


@app.get("/api/audio/{trigger_id}/{action_index}")
async def get_audio(
    trigger_id: str, action_index: int, manager: ManagerDep
) -> Response:
    """Fetch rendered audio for a speak or think action.

    Returns:
        - 200 with audio file if ready
        - 202 Accepted if still rendering/pending or just queued
        - 404 if not found or not a speak/think action
        - 503 if TTS service not available
    """
    agent: Agent = manager.agent

    if agent.tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not available")

    action_id = f"{trigger_id}_{action_index}"
    status = agent.tts_service.get_audio_status(action_id)

    if status == RenderStatus.READY:
        audio_path = agent.tts_service.get_audio_path(action_id)
        if audio_path is not None and audio_path.exists():
            return FileResponse(
                path=str(audio_path),
                media_type="audio/mpeg",
                filename=f"{action_id}.mp3",
            )
        # File not found despite status being ready - treat as error
        return Response(status_code=404)

    elif status in (RenderStatus.PENDING, RenderStatus.RENDERING):
        # Not ready yet - client should retry
        return Response(status_code=202)

    else:
        # ERROR status - try to queue on-demand rendering
        if _try_queue_tts_render(agent, trigger_id, action_index):
            return Response(status_code=202)  # Queued, client should poll
        return Response(status_code=404)  # Not a speak/think action or not found


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

    # Mount generated audio directory
    app.mount(
        "/generated_audio",
        StaticFiles(directory=agent_paths.get_generated_audio_dir()),
        name="generated_audio",
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
