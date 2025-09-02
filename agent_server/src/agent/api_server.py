"""
FastAPI server for single-user agent system
"""

import json
import logging
import os

from agent.types import Message
from agent.llm import create_llm, SupportedModel
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.core import Agent
from agent.paths import agent_paths
from agent.api_types import (
    TriggerHistoryResponse,
    TimelineResponse,
    TimelineEntry,
    TimelineEntryTrigger,
    TimelineEntrySummary,
    PaginationInfo,
    convert_trigger_history_entry_to_dto,
    convert_summary_to_dto,
)
from agent.conversation_persistence import AgentData
from agent.state import State
from agent.chain_of_action.trigger_history import TriggerHistory


# Response Models
class ConversationResponse(BaseModel):
    messages: List[Message]


class ResetResponse(BaseModel):
    message: str
    timestamp: str


class AutoWakeupStatusResponse(BaseModel):
    enabled: bool
    delay_seconds: int


class AutoWakeupSetRequest(BaseModel):
    enabled: bool


class AutoWakeupSetResponse(BaseModel):
    enabled: bool
    message: str
    timestamp: str


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def initialize_agent(load: bool) -> Agent:
    """Initialize the agent with specific conversation files for development"""
    llm = create_llm()
    agent = Agent(
        model=SupportedModel.MISTRAL_SMALL_3_2_Q4,
        llm=llm,
        enable_image_generation=True,
        individual_trigger_compression=True,
        auto_save=True,
    )

    if load:
        try:
            agent.load_conversation("baseline")
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")

    return agent


app = FastAPI(
    title="Agent API",
    description="Single-User Streaming AI Agent API",
    version="1.0.0",
)

app.state.agent = initialize_agent(
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
    context_info = app.state.agent.get_context_info()
    return {
        "message_count": context_info.message_count,
        "conversation_messages": context_info.conversation_messages,
        "estimated_tokens": context_info.estimated_tokens,
        "context_limit": context_info.context_limit,
        "usage_percentage": context_info.usage_percentage,
        "approaching_limit": context_info.approaching_limit,
    }


@app.get("/api/trigger-history", response_model=TriggerHistoryResponse)
async def get_trigger_history():
    """Get current trigger history for client hydration"""
    agent: Agent = app.state.agent
    trigger_history = agent.get_trigger_history()

    # Convert all entries and summaries to DTOs
    entry_dtos = [
        convert_trigger_history_entry_to_dto(entry)
        for entry in trigger_history.get_all_entries()
    ]

    # Include initial exchange if it exists (for UI display)
    if agent.initial_exchange is not None:
        initial_exchange_dto = convert_trigger_history_entry_to_dto(
            agent.initial_exchange
        )
        entry_dtos.insert(0, initial_exchange_dto)  # Insert at beginning

    summary_dtos = [
        convert_summary_to_dto(summary)
        for summary in trigger_history.get_all_summaries()
    ]

    recent_entries = trigger_history.get_recent_entries()

    return TriggerHistoryResponse(
        entries=entry_dtos,
        summaries=summary_dtos,
        total_entries=len(trigger_history),
        recent_entries_count=len(recent_entries),
    )


@app.get("/api/timeline", response_model=TimelineResponse)
async def get_timeline(
    page_size: int = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    """Get paginated timeline in chronological order, defaulting to most recent page"""
    agent: Agent = app.state.agent
    trigger_history = agent.get_trigger_history()

    # Build timeline entries in proper chronological order with summaries interspersed
    timeline_entries: List[TimelineEntry] = []

    # Get all the data we need
    entries = trigger_history.get_all_entries()
    summaries = trigger_history.get_all_summaries()

    # Create a map of insert positions to summaries for efficient lookup
    summary_map = {summary.insert_at_index: summary for summary in summaries}

    # Current position in the timeline (0-based to match insert_at_index)
    current_position = 0

    # Add initial exchange if it exists (position 0)
    if agent.initial_exchange is not None:
        initial_dto = convert_trigger_history_entry_to_dto(agent.initial_exchange)
        timeline_entries.append(TimelineEntryTrigger(entry=initial_dto))
        current_position += 1

    # Process trigger entries and insert summaries at correct positions
    for entry in entries:
        # Check if there's a summary that should be inserted at this position
        if current_position in summary_map:
            summary = summary_map[current_position]
            summary_dto = convert_summary_to_dto(summary)
            timeline_entries.append(TimelineEntrySummary(summary=summary_dto))
            current_position += 1

        # Add the trigger entry
        entry_dto = convert_trigger_history_entry_to_dto(entry)
        timeline_entries.append(TimelineEntryTrigger(entry=entry_dto))
        current_position += 1

    # Check for any remaining summaries that should be at the end
    while current_position in summary_map:
        summary = summary_map[current_position]
        summary_dto = convert_summary_to_dto(summary)
        timeline_entries.append(TimelineEntrySummary(summary=summary_dto))
        current_position += 1

    total_items = len(timeline_entries)

    # Debug pagination logic
    logger.info(
        f"Timeline request: page_size={page_size}, after={after}, before={before}, total_items={total_items}"
    )

    # Handle pagination with after/before cursors
    if before is not None:
        # Get entries before the specified index (older entries)
        try:
            before_index = int(before)
            end_index = min(before_index, total_items)
            start_index = max(0, end_index - page_size)
        except (ValueError, TypeError):
            # Invalid before cursor, default to last page
            start_index = max(0, total_items - page_size)
            end_index = total_items
    elif after is not None:
        # Get entries after the specified index (newer entries)
        try:
            after_index = int(after)
            start_index = min(after_index, total_items)
            end_index = min(start_index + page_size, total_items)
        except (ValueError, TypeError):
            # Invalid after cursor, default to last page
            start_index = max(0, total_items - page_size)
            end_index = total_items
    else:
        # Default to showing the last page (most recent items)
        start_index = max(0, total_items - page_size)
        end_index = total_items

    # Get page of items
    page_entries = timeline_entries[start_index:end_index]

    # Calculate pagination info
    has_next = end_index < total_items  # Can go forward to newer items
    has_previous = start_index > 0  # Can go back to older items

    # Next cursor (after): entries after the current page end
    next_cursor = str(end_index) if has_next else None

    # Previous cursor (before): entries before the current page start
    previous_cursor = str(start_index) if has_previous else None

    pagination = PaginationInfo(
        total_items=total_items,
        page_size=page_size,
        has_next=has_next,
        has_previous=has_previous,
        next_cursor=next_cursor,
        previous_cursor=previous_cursor,
    )

    return TimelineResponse(
        entries=page_entries,
        pagination=pagination,
    )


@app.post("/api/reset", response_model=ResetResponse)
async def reset_agent():
    """Reset the agent"""

    # Get the old agent to transfer state and clean up resources
    old_agent: Agent | None = app.state.agent
    current_client_queue = None

    if old_agent:
        # Disable wakeup timer scheduling and cancel any active timer
        old_agent.set_auto_wakeup_enabled(False)

        # Disable auto-save to prevent the old agent from saving mid-reset
        old_agent.auto_save = False

        # Get the current client queue to transfer to new agent
        with old_agent.client_queue_lock:
            current_client_queue = old_agent.current_client_queue
            # Clear the old agent's queue reference so it stops pushing events
            old_agent.current_client_queue = None

    # Reinitialize the agent
    new_agent = initialize_agent(
        load=False  # Set to False to avoid loading specific conversation
    )

    # Transfer the client queue to the new agent if one exists
    if current_client_queue is not None:
        # Clear any remaining events from the old agent before transferring
        while not current_client_queue.empty():
            try:
                current_client_queue.get_nowait()
            except:
                break

        new_agent.set_client_queue(current_client_queue)

    app.state.agent = new_agent

    return ResetResponse(
        message="Agent reset successfully",
        timestamp=datetime.now().isoformat(),
    )


@app.websocket("/api/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()

    import asyncio
    import threading
    import queue as queue_module

    # Create client-specific queue and register with agent (replaces any existing client)
    client_queue: queue_module.Queue = queue_module.Queue()
    app.state.agent.set_client_queue(client_queue)

    logger.info("WebSocket client connected, queue registered")

    try:

        async def handle_incoming_messages():
            """Handle incoming messages from client"""
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    message = message_data.get("message", "")

                    # Handle empty message as wakeup trigger
                    if not message.strip():
                        # Create wakeup trigger instead of user input trigger
                        message = None  # Signal to use wakeup trigger

                    # Process message in background thread
                    logger.info(f"Processing message: {message}")

                    def process_message():
                        try:
                            app.state.agent.chat_stream(message)
                        except Exception as e:
                            # Put error event in queue
                            from agent.api_types import AgentErrorEvent

                            error_event = AgentErrorEvent(
                                message=f"Internal error: {str(e)}"
                            )
                            app.state.agent.emit_event(error_event)

                    # Run agent processing in background thread
                    thread = threading.Thread(target=process_message)
                    thread.start()
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")

        async def handle_outgoing_events():
            """Handle outgoing events to client"""
            try:
                while True:
                    # Get event from our local client queue with timeout
                    try:
                        event = await asyncio.to_thread(client_queue.get, True, 1.0)
                        await websocket.send_text(event.model_dump_json())
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
        # Clear our queue from the agent (only if it's still ours)
        app.state.agent.clear_client_queue(client_queue)
        logger.info("WebSocket client disconnected, queue cleared")


@app.get("/api/auto-wakeup", response_model=AutoWakeupStatusResponse)
async def get_auto_wakeup_status():
    """Get current auto-wakeup status"""
    agent: Agent = app.state.agent
    return AutoWakeupStatusResponse(
        enabled=agent.get_auto_wakeup_enabled(),
        delay_seconds=agent.wakeup_delay_seconds,
    )


@app.post("/api/auto-wakeup", response_model=AutoWakeupSetResponse)
async def set_auto_wakeup_status(request: AutoWakeupSetRequest):
    """Set auto-wakeup enabled state"""
    agent: Agent = app.state.agent

    agent.set_auto_wakeup_enabled(request.enabled)

    return AutoWakeupSetResponse(
        enabled=agent.get_auto_wakeup_enabled(),
        message=f"Auto-wakeup {'enabled' if request.enabled else 'disabled'}",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "agent_initialized": app.state.agent is not None,
        "agent_name": app.state.agent.state.name,
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

    # Catch-all route for React SPA (must be last!)
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve React SPA, fallback to index.html for client-side routing"""

        logger.info(f"Serving SPA for path: {path}")

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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
