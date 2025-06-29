"""
FastAPI server for single-user agent system
"""

from enum import Enum
import json
import logging
from agent.message import Message
from typing import Dict, Any, List, Literal, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path

from agent.core import Agent
from agent.config import get_config as get_agent_config, get_all_configs
from agent.agent_events import (
    AgentEvent,
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)


# Response Models
class ConversationResponse(BaseModel):
    messages: List[Message]


class ConfigResponse(BaseModel):
    name: str
    description: str
    tools: List[str]


class ConfigsResponse(BaseModel):
    configs: Dict[str, str]  # name -> description


class ResetResponse(BaseModel):
    message: str
    config: str
    timestamp: str


# Set up logging
logger = logging.getLogger(__name__)

# Global single-user agent instance
agent: Agent = None


app = FastAPI(
    title="Agent API", description="Single-User Streaming AI Agent API", version="1.0.0"
)

# Add CORS middleware for local network access (phone, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from any device on local network
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files configuration
CLIENT_DIST_PATH = Path(__file__).parent.parent.parent / "client" / "dist"

def setup_static_files():
    """Setup static file serving for the React client"""
    if CLIENT_DIST_PATH.exists():
        print(f"✅ Serving React client from: {CLIENT_DIST_PATH}")
        
        # Mount static assets (JS, CSS, images, etc.)
        app.mount("/assets", StaticFiles(directory=CLIENT_DIST_PATH / "assets"), name="assets")
        
        # Catch-all route for React SPA (must be last!)
        @app.get("/{path:path}")
        async def serve_spa(path: str):
            """Serve React SPA, fallback to index.html for client-side routing"""
            
            # Try to serve specific file first
            file_path = CLIENT_DIST_PATH / path
            if file_path.is_file():
                return FileResponse(file_path)
            
            # Fallback to index.html for SPA routing
            index_path = CLIENT_DIST_PATH / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            
            return {"message": "React client not built. Run 'cd client && npm run build'"}
        
    else:
        print(f"⚠️  React client not found at: {CLIENT_DIST_PATH}")
        print("   Run 'cd client && npm run build' to build the client first")
        
        # Provide helpful message at root
        @app.get("/")
        async def no_client():
            return {
                "message": "Agent API Server", 
                "client_status": "not_built",
                "instructions": "Run 'cd client && npm run build' to enable web interface"
            }


def initialize_agent(config_name: str = "roleplay") -> Agent:
    """Initialize the global agent instance"""
    global agent
    config = get_agent_config(config_name)
    agent = Agent(
        config=config,
        model="huihui_ai/mistral-small-abliterated",
        verbose=False,
    )
    return agent


def get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot notation"""
    keys = path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(f"Key '{key}' not found")
            current = current[key]
        elif isinstance(current, list):
            try:
                index = int(key)
                current = current[index]
            except (ValueError, IndexError):
                raise KeyError(f"Invalid list index '{key}'")
        else:
            raise KeyError(f"Cannot navigate through non-dict/list at '{key}'")

    return current


def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set value in nested dict using dot notation"""
    keys = path.split(".")
    current = data

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if isinstance(current, dict):
            if key not in current:
                current[key] = {}  # Create missing intermediate dicts
            current = current[key]
        elif isinstance(current, list):
            try:
                index = int(key)
                current = current[index]
            except (ValueError, IndexError):
                raise KeyError(f"Invalid list index '{key}'")
        else:
            raise KeyError(f"Cannot navigate through non-dict/list at '{key}'")

    # Set the final value
    final_key = keys[-1]
    if isinstance(current, dict):
        current[final_key] = value
    elif isinstance(current, list):
        try:
            index = int(final_key)
            current[index] = value
        except (ValueError, IndexError):
            raise KeyError(f"Invalid list index '{final_key}'")
    else:
        raise KeyError(f"Cannot set value on non-dict/list")


def event_to_dict(event: AgentEvent) -> Dict[str, Any]:
    """Convert agent event to dictionary for JSON serialization"""
    base_dict = {"type": event.type.value}

    if isinstance(event, AgentTextEvent):
        base_dict.update({"content": event.content})
    elif isinstance(event, ToolStartedEvent):
        base_dict.update(
            {
                "tool_name": event.tool_name,
                "tool_id": event.tool_id,
                "parameters": event.parameters,
            }
        )
    elif isinstance(event, ToolFinishedEvent):
        base_dict.update(
            {
                "tool_id": event.tool_id,
                "result_type": event.result_type.value,
                "result": event.result,
            }
        )
    elif isinstance(event, AgentErrorEvent):
        base_dict.update(
            {
                "message": event.message,
                "tool_name": event.tool_name,
                "tool_id": event.tool_id,
            }
        )

    return base_dict


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    logging.basicConfig(level=logging.INFO)
    initialize_agent()
    setup_static_files()


@app.get("/api/conversation", response_model=ConversationResponse)
async def get_conversation():
    """Get current conversation history for client hydration"""
    if agent is None:
        initialize_agent()

    # Return the structured conversation history
    messages = agent.get_conversation_history()
    return ConversationResponse(messages=messages)


@app.get("/api/state")
async def get_state(path: Optional[str] = None):
    """Get agent state, optionally by dot notation path"""
    if agent is None:
        initialize_agent()

    state = agent.get_state()

    if path:
        try:
            return get_nested_value(state, path)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return state


@app.put("/api/state")
async def set_state(request: Request, path: Optional[str] = None):
    """Set agent state, optionally by dot notation path"""
    if agent is None:
        initialize_agent()

    # Get request body as JSON
    body = await request.json()

    if path:
        # Set specific path
        try:
            current_state = agent.get_state()
            set_nested_value(current_state, path, body)
            # Note: agent.state is updated by reference
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"message": f"Updated state at path: {path}"}
    else:
        # Replace entire state
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="State must be a dictionary")

        # Replace agent state entirely
        agent.state = body
        return {"message": "State replaced successfully"}


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current agent configuration info"""
    if agent is None:
        initialize_agent()

    config = agent.config
    tool_names = [tool.name for tool in config.tools]

    return ConfigResponse(
        name=config.name, description=config.description, tools=tool_names
    )


@app.get("/api/configs", response_model=ConfigsResponse)
async def get_configs():
    """Get all available agent configurations"""
    configs = {}
    for name, config in get_all_configs().items():
        configs[name] = config.description

    return ConfigsResponse(configs=configs)


@app.post("/api/reset", response_model=ResetResponse)
async def reset_agent(config: str = "roleplay"):
    """Reset agent with specified configuration"""
    global agent

    available_configs = get_all_configs()
    if config not in available_configs:
        available = list(available_configs.keys())
        raise HTTPException(
            status_code=400, detail=f"Invalid config '{config}'. Available: {available}"
        )

    # Reinitialize agent with specified config
    initialize_agent(config)

    return ResetResponse(
        message="Agent reset successfully",
        config=config,
        timestamp=datetime.now().isoformat(),
    )


@app.websocket("/api/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()

    if agent is None:
        initialize_agent()

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
                for event in agent.chat_stream(message):
                    event_dict = event_to_dict(event)
                    logger.debug(f"Sending event: {event_dict}")
                    await websocket.send_text(json.dumps(event_dict))

                # Send end-of-response marker
                await websocket.send_text(json.dumps({"type": "response_complete"}))

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
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "agent_config": agent.config.name if agent else None,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
