"""
Pytest test suite for API endpoints with automatic server management
"""

import asyncio
import json
import pytest
import requests
import websockets
import subprocess
import time
import os
from typing import Generator

# Test configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/chat"
SERVER_STARTUP_TIMEOUT = 15


@pytest.fixture(scope="session")
def api_server() -> Generator[subprocess.Popen, None, None]:
    """Start API server for testing session"""
    print("\n🚀 Starting API server for tests...")

    # Set up environment
    env = os.environ.copy()

    # Start server process
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "agent.api_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--app-dir",
            "src",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    server_started = False
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=1)
            if response.status_code == 200:
                print(f"✅ Server started (took {i+1}s)")
                server_started = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    if not server_started:
        process.terminate()
        pytest.fail("Server failed to start within timeout")

    yield process

    # Teardown
    print("\n🛑 Stopping API server...")
    try:
        process.terminate()
        process.wait(timeout=5)
        print("✅ Server stopped")
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print("✅ Server killed")

    # Disaply server logs
    print("\n📜 Server logs")
    stdout, stderr = process.communicate()
    if stdout:
        print("\nSTDOUT:\n", stdout.decode())
    if stderr:
        print("\nSTDERR:\n", stderr.decode())


@pytest.fixture
def reset_agent(api_server):
    """Reset agent to roleplay config before each test"""
    requests.post(f"{BASE_URL}/reset?config=roleplay", timeout=5)
    yield
    # Cleanup after test if needed


class TestHealthEndpoint:
    """Test health check functionality"""

    @pytest.mark.integration
    def test_health_check(self, api_server):
        """Test basic health endpoint"""
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(response.text)

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "agent_initialized" in data
        assert "agent_config" in data
        assert "timestamp" in data

        # Check values
        assert data["status"] == "healthy"
        assert data["agent_initialized"] is True
        assert isinstance(data["agent_config"], str)


class TestConfigEndpoints:
    """Test configuration management"""

    @pytest.mark.integration
    def test_get_current_config(self, api_server, reset_agent):
        """Test getting current config"""
        response = requests.get(f"{BASE_URL}/api/config", timeout=5)

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "name" in data
        assert "description" in data
        assert "tools" in data

        # Check types
        assert isinstance(data["name"], str)
        assert isinstance(data["description"], str)
        assert isinstance(data["tools"], list)

    @pytest.mark.integration
    def test_get_available_configs(self, api_server):
        """Test getting all available configs"""
        response = requests.get(f"{BASE_URL}/api/configs", timeout=5)

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "configs" in data
        assert isinstance(data["configs"], dict)

        # Check expected configs exist
        expected_configs = ["roleplay", "coding", "general"]
        for config_name in expected_configs:
            assert config_name in data["configs"]
            assert isinstance(data["configs"][config_name], str)

    @pytest.mark.integration
    def test_reset_with_config_change(self, api_server, reset_agent):
        """Test resetting agent with different config"""
        # Get current config
        current_resp = requests.get(f"{BASE_URL}/api/config", timeout=5)
        current_config = current_resp.json()["name"]

        # Reset to different config
        new_config = "coding" if current_config != "coding" else "general"
        reset_resp = requests.post(
            f"{BASE_URL}/api/reset?config={new_config}", timeout=5
        )

        assert reset_resp.status_code == 200
        reset_data = reset_resp.json()

        assert "message" in reset_data
        assert reset_data["config"] == new_config

        # Verify config actually changed
        new_resp = requests.get(f"{BASE_URL}/api/config", timeout=5)
        new_config_data = new_resp.json()

        assert new_config_data["name"] == new_config

    @pytest.mark.integration
    def test_reset_with_invalid_config(self, api_server):
        """Test reset with invalid config returns error"""
        response = requests.post(
            f"{BASE_URL}/api/reset?config=invalid_config", timeout=5
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "invalid_config" in data["detail"].lower()


class TestConversationEndpoint:
    """Test conversation history"""

    @pytest.mark.integration
    def test_get_empty_conversation(self, api_server, reset_agent):
        """Test getting conversation when empty"""
        response = requests.get(f"{BASE_URL}/api/conversation", timeout=5)

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "messages" in data

        # Check empty state
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 0


class TestStateEndpoints:
    """Test state management"""

    @pytest.mark.integration
    def test_get_full_state(self, api_server, reset_agent):
        """Test getting full agent state"""
        # Set some initial state
        initial_state = {
            "current_character_id": None,
            "characters": {},
            "global_scene": None,
            "global_memories": [],
        }
        response = requests.put(
            f"{BASE_URL}/api/state",
            json=initial_state,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )

        response = requests.get(f"{BASE_URL}/api/state", timeout=5)

        assert response.status_code == 200
        data = response.json()

        # Check it's a dictionary
        assert isinstance(data, dict)

        # For roleplay config, expect these keys
        expected_keys = [
            "current_character_id",
            "characters",
            "global_scene",
            "global_memories",
        ]
        for key in expected_keys:
            assert key in data

    @pytest.mark.integration
    def test_get_state_by_path(self, api_server, reset_agent):
        """Test getting state by dot notation path"""
        # Get specific path
        response = requests.get(
            f"{BASE_URL}/api/state?path=current_character_id", timeout=5
        )

        assert response.status_code == 404

    @pytest.mark.integration
    def test_set_state_by_path(self, api_server, reset_agent):
        """Test setting state by dot notation path"""
        test_value = "test_character_pytest"

        # Set value
        put_response = requests.put(
            f"{BASE_URL}/api/state?path=current_character_id",
            json=test_value,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )

        assert put_response.status_code == 200
        put_data = put_response.json()
        assert "current_character_id" in put_data

        # Verify value was set
        get_response = requests.get(
            f"{BASE_URL}/api/state?path=current_character_id", timeout=5
        )
        assert get_response.status_code == 200
        assert get_response.json() == test_value

    @pytest.mark.integration
    def test_get_invalid_state_path(self, api_server, reset_agent):
        """Test getting non-existent state path returns 404"""
        response = requests.get(f"{BASE_URL}/api/state?path=nonexistent.key", timeout=5)

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    def test_set_nested_state_path_creates_intermediates(self, api_server, reset_agent):
        """Test setting nested state path creates intermediate dictionaries"""
        test_value = "deeply_nested_value"

        # Set a deeply nested path that doesn't exist
        response = requests.put(
            f"{BASE_URL}/api/state?path=new_section.subsection.key",
            json=test_value,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )

        assert response.status_code == 200

        # Verify the value was set and intermediates were created
        get_response = requests.get(
            f"{BASE_URL}/api/state?path=new_section.subsection.key", timeout=5
        )
        assert get_response.status_code == 200
        assert get_response.json() == test_value

        # Verify intermediate dict was created
        intermediate_response = requests.get(
            f"{BASE_URL}/api/state?path=new_section", timeout=5
        )
        assert intermediate_response.status_code == 200
        intermediate_data = intermediate_response.json()
        assert isinstance(intermediate_data, dict)
        assert "subsection" in intermediate_data


class TestGeneratedImageEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generated_image_endpoint(self, api_server, reset_agent):
        images = requests.get(f"{BASE_URL}/api/generated_images", timeout=5)

        assert images.status_code == 200
        data = images.json()

        # Check response structure
        assert isinstance(data, list)
        for item in data:
            assert "image_url" in item
            assert isinstance(item["image_url"], str)
            assert item["image_url"].startswith("http://") or item[
                "image_url"
            ].startswith("https://")


class TestWebSocketChat:
    """Test WebSocket streaming chat (single-user system)"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_simple_chat(self, api_server, reset_agent):
        """Test basic WebSocket chat functionality"""
        async with websockets.connect(WS_URL) as websocket:
            # Send test message
            test_message = {"message": "Just say 'Hello' and nothing else"}
            await websocket.send(json.dumps(test_message))

            # Collect responses
            events = []
            response_complete = False
            timeout_count = 0
            max_timeouts = 6  # 30 seconds total

            while not response_complete and timeout_count < max_timeouts:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    event = json.loads(response)
                    events.append(event)

                    if event.get("type") == "response_complete":
                        response_complete = True
                    elif event.get("type") == "agent_error":
                        pytest.fail(f"Agent error: {event.get('message')}")

                except asyncio.TimeoutError:
                    timeout_count += 1
                    continue

            # Verify we got a complete response
            assert (
                response_complete
            ), f"Response not completed. Got {len(events)} events"

            # Check we got text events
            text_events = [e for e in events if e.get("type") == "text"]
            assert len(text_events) > 0, "No text events received"

            # Verify text content exists
            full_text = "".join(e.get("content", "") for e in text_events)
            assert len(full_text.strip()) > 0, "Empty response text"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_websocket_connection_lifecycle(self, api_server, reset_agent):
        """Test WebSocket connection opens and closes properly"""
        # Test single connection lifecycle
        async with websockets.connect(WS_URL) as websocket:
            # Send minimal message
            await websocket.send(json.dumps({"message": "Hi"}))

            # Just verify we can receive something
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                event = json.loads(response)
                assert "type" in event
            except asyncio.TimeoutError:
                pytest.fail("Connection timed out")

        # Connection should be closed cleanly after context exit


# Integration test
class TestFullWorkflow:
    """Test complete API workflow"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_roleplay_workflow(self, api_server, reset_agent):
        """Test a complete roleplay interaction workflow"""
        # Reset the agent to ensure clean state
        requests.post(f"{BASE_URL}/reset?config=roleplay", timeout=5)

        # 1. Verify initial state
        state_resp = requests.get(f"{BASE_URL}/api/state", timeout=5)
        assert state_resp.status_code == 200
        initial_state = state_resp.json()
        assert initial_state["current_character_id"] is None

        # 2. Send character creation message via WebSocket
        async with websockets.connect(WS_URL) as websocket:
            character_message = {
                "message": "Please assume the character Luna, a friendly cat with a playful personality"
            }
            await websocket.send(json.dumps(character_message))

            # Collect response
            events = []
            response_complete = False
            timeout_count = 0

            while not response_complete and timeout_count < 6:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    event = json.loads(response)
                    events.append(event)

                    if event.get("type") == "response_complete":
                        response_complete = True
                    elif event.get("type") == "agent_error":
                        pytest.fail(
                            f"Character creation failed: {event.get('message')}"
                        )

                except asyncio.TimeoutError:
                    timeout_count += 1

            assert response_complete, "Character creation did not complete"

        # 3. Verify conversation history has messages
        conv_resp = requests.get(f"{BASE_URL}/api/conversation", timeout=5)
        conversation = conv_resp.json()

        assert (
            len(conversation["messages"]) >= 2
        )  # At least user message + assistant response

        # 4. Test state persistence across requests
        # Make another request and verify conversation continues
        async with websockets.connect(WS_URL) as websocket:
            follow_up = {"message": "What's your name?"}
            await websocket.send(json.dumps(follow_up))

            # Just verify we get a response
            response_complete = False
            timeout_count = 0

            while not response_complete and timeout_count < 6:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    event = json.loads(response)

                    if event.get("type") == "response_complete":
                        response_complete = True
                    elif event.get("type") == "agent_error":
                        pytest.fail(f"Follow-up failed: {event.get('message')}")

                except asyncio.TimeoutError:
                    timeout_count += 1

            assert response_complete, "Follow-up conversation did not complete"

        # 5. Verify conversation history grew
        final_conv_resp = requests.get(f"{BASE_URL}/api/conversation", timeout=5)
        final_conversation = final_conv_resp.json()

        assert len(final_conversation["messages"]) >= 4  # Original + follow-up


if __name__ == "__main__":
    # Run with pytest command instead
    print("Run with: pytest tests/test_api.py -v")
