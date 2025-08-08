"""
FastAPI server for single-user agent system
"""

import json
import logging

from agent.types import Message
from agent.llm import create_llm, SupportedModel
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.core import Agent
from agent.paths import agent_paths
from agent.api_types import (
    TriggerHistoryResponse,
    convert_trigger_history_entry_to_dto,
    convert_summary_to_dto,
)


# Response Models
class ConversationResponse(BaseModel):
    messages: List[Message]


class ResetResponse(BaseModel):
    message: str
    timestamp: str


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def initialize_agent() -> Agent:
    """Initialize the agent

    Args:
        load_conversation: Whether to load conversation.json up to first summary
    """
    llm = create_llm()
    agent = Agent(
        model=SupportedModel.MISTRAL_SMALL_3_2,
        llm=llm,
        enable_image_generation=True,
    )

    return agent


app = FastAPI(
    title="Agent API",
    description="Single-User Streaming AI Agent API",
    version="1.0.0",
)

app.state.agent = initialize_agent(
    # load_conversation=True
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


@app.post("/api/reset", response_model=ResetResponse)
async def reset_agent():
    """Reset the agent"""

    # Reinitialize the agent
    app.state.agent = initialize_agent()

    return ResetResponse(
        message="Agent reset successfully",
        timestamp=datetime.now().isoformat(),
    )


@app.websocket("/api/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")

            if not message.strip():
                continue

            # Stream agent response
            try:
                logger.info(f"Processing message: {message}")
                for event in app.state.agent.chat_stream(message):
                    await websocket.send_text(event.model_dump_json())

                # Agent will emit ResponseCompleteEvent automatically

            except Exception as e:
                # Send error event
                error_event = {
                    "type": "agent_error",
                    "message": f"Internal error: {str(e)}",
                    "tool_name": None,
                    "tool_id": None,
                }
                await websocket.send_text(json.dumps(error_event))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


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
