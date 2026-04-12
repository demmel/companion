from types import SimpleNamespace
from pathlib import Path

import ollama
from fastapi.testclient import TestClient

from agent.api_server import app as production_app
from agent.api_server import create_app
from agent.config import Config
from agent.llm.models import ModelConfig, SupportedModel


class DummyManager:
    def __init__(self, name: str = "Chloe") -> None:
        self.state = SimpleNamespace(name=name)
        self.agent = SimpleNamespace(close=lambda: None)


class DummyOllamaClient:
    def __init__(
        self,
        *,
        models=None,
        list_error: Exception | None = None,
        pull_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self._models = models or []
        self._list_error = list_error
        self._pull_error = pull_error
        self._delete_error = delete_error
        self.pulled: list[str] = []
        self.deleted: list[str] = []

    def list(self):
        if self._list_error is not None:
            raise self._list_error
        return ollama._types.ListResponse.model_validate({"models": self._models})

    def pull(self, name: str):
        if self._pull_error is not None:
            raise self._pull_error
        self.pulled.append(name)
        return {"status": "success"}

    def delete(self, name: str):
        if self._delete_error is not None:
            raise self._delete_error
        self.deleted.append(name)
        return {"status": "success"}


def test_health_endpoint_reports_healthy_when_agent_starts():
    app = create_app(agent_manager_factory=lambda: DummyManager())
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["agent_name"] == "Chloe"


def test_health_endpoint_reports_startup_failure_without_crashing_server():
    app = create_app(initialize_on_startup=False)
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        app.state.agent_startup_error = "missing llm backend"
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["startup_error"] == "missing llm backend"


def test_api_routes_return_503_when_agent_is_unavailable():
    app = create_app(initialize_on_startup=False)
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.get("/api/context")

    assert response.status_code == 503
    assert response.json()["detail"] == "Agent is unavailable"


def test_get_installed_ollama_models_returns_sorted_models():
    ollama_client = DummyOllamaClient(
        models=[
            {
                "model": "zeta:latest",
                "size": 42,
                "modified_at": "2026-04-10T12:00:00Z",
                "details": {},
            },
            {
                "model": "alpha:latest",
                "size": 7,
                "digest": "abc123",
                "details": {},
            },
        ]
    )
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: ollama_client,
    )
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.get("/api/ollama/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "name": "alpha:latest",
                "size": 7,
                "modified_at": None,
                "digest": "abc123",
                "details": {},
            },
            {
                "name": "zeta:latest",
                "size": 42,
                "modified_at": "2026-04-10T12:00:00+00:00",
                "digest": None,
                "details": {},
            },
        ]
    }


def test_get_supported_models_returns_installed_ollama_models():
    ollama_client = DummyOllamaClient(
        models=[
            {
                "model": "installed-model:latest",
                "size": 100,
                "details": {},
            },
        ]
    )
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: ollama_client,
    )
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.get("/api/supported-models")

    assert response.status_code == 200
    payload = response.json()
    assert "installed-model:latest" in payload["ollama_models"]
    assert "claude-sonnet-4-5-20250929" in payload["models"]
    assert "claude-sonnet-4-5-20250929" not in payload["ollama_models"]


def test_get_supported_models_falls_back_to_known_ollama_models_when_ollama_unavailable():
    ollama_client = DummyOllamaClient(list_error=RuntimeError("connection refused"))
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: ollama_client,
    )
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.get("/api/supported-models")

    assert response.status_code == 200
    payload = response.json()
    # Falls back to hardcoded known ollama model suggestions
    assert "mistral-small3.2:latest" in payload["ollama_models"]
    assert "claude-sonnet-4-5-20250929" in payload["models"]
    assert "claude-sonnet-4-5-20250929" not in payload["ollama_models"]


def test_update_model_config_accepts_arbitrary_ollama_model_names(tmp_path: Path):
    original_model_config = Config._model_config
    original_model_config_path = Config._model_config_path
    original_agent_manager = production_app.state.agent_manager
    original_startup_error = production_app.state.agent_startup_error
    Config._model_config = None
    Config._model_config_path = tmp_path / "model_config.json"
    production_app.state.agent_manager = DummyManager()
    production_app.state.agent_startup_error = None

    payload = {
        "state_initialization_model": "custom-org/custom-reasoner:latest",
        "action_planning_model": "claude-sonnet-4-5-20250929",
        "situational_analysis_model": "claude-sonnet-4-5-20250929",
        "memory_retrieval_model": "claude-sonnet-4-5-20250929",
        "memory_formation_model": "claude-sonnet-4-5-20250929",
        "trigger_compression_model": "claude-sonnet-4-5-20250929",
        "think_action_model": "custom-org/custom-reasoner:latest",
        "speak_action_model": "custom-org/custom-reasoner:latest",
        "visual_action_model": "custom-org/custom-reasoner:latest",
        "fetch_url_action_model": "claude-sonnet-4-5-20250929",
        "evaluate_priorities_action_model": "claude-sonnet-4-5-20250929",
        "tts_rewrite_model": "custom-org/custom-reasoner:latest",
    }

    try:
        with TestClient(production_app) as client:
            response = client.post("/api/model-config", json=payload)
            follow_up = client.get("/api/model-config")
    finally:
        Config._model_config = original_model_config
        Config._model_config_path = original_model_config_path
        production_app.state.agent_manager = original_agent_manager
        production_app.state.agent_startup_error = original_startup_error

    assert response.status_code == 200
    assert response.json()["config"]["state_initialization_model"] == payload["state_initialization_model"]
    assert follow_up.status_code == 200
    assert follow_up.json()["think_action_model"] == payload["think_action_model"]


def test_update_model_config_rejects_unknown_claude_model_names(tmp_path: Path):
    original_model_config = Config._model_config
    original_model_config_path = Config._model_config_path
    original_agent_manager = production_app.state.agent_manager
    original_startup_error = production_app.state.agent_startup_error
    Config._model_config = None
    Config._model_config_path = tmp_path / "model_config.json"
    production_app.state.agent_manager = DummyManager()
    production_app.state.agent_startup_error = None

    payload = {
        "state_initialization_model": "claude-unknown-next",
        "action_planning_model": "claude-sonnet-4-5-20250929",
        "situational_analysis_model": "claude-sonnet-4-5-20250929",
        "memory_retrieval_model": "claude-sonnet-4-5-20250929",
        "memory_formation_model": "claude-sonnet-4-5-20250929",
        "trigger_compression_model": "claude-sonnet-4-5-20250929",
        "think_action_model": "claude-sonnet-4-5-20250929",
        "speak_action_model": "claude-sonnet-4-5-20250929",
        "visual_action_model": "claude-sonnet-4-5-20250929",
        "fetch_url_action_model": "claude-sonnet-4-5-20250929",
        "evaluate_priorities_action_model": "claude-sonnet-4-5-20250929",
        "tts_rewrite_model": "mistral-small3.2:latest",
    }

    try:
        with TestClient(production_app) as client:
            response = client.post("/api/model-config", json=payload)
    finally:
        Config._model_config = original_model_config
        Config._model_config_path = original_model_config_path
        production_app.state.agent_manager = original_agent_manager
        production_app.state.agent_startup_error = original_startup_error

    assert response.status_code == 400
    assert "not a known Anthropic model" in response.json()["detail"]


def test_update_model_config_rejects_whitespace_padded_claude_model_names(tmp_path: Path):
    original_model_config = Config._model_config
    original_model_config_path = Config._model_config_path
    original_agent_manager = production_app.state.agent_manager
    original_startup_error = production_app.state.agent_startup_error
    Config._model_config = None
    Config._model_config_path = tmp_path / "model_config.json"
    production_app.state.agent_manager = DummyManager()
    production_app.state.agent_startup_error = None

    payload = {
        "state_initialization_model": " claude-unknown-next",  # leading space bypasses naive prefix check
        "action_planning_model": "claude-sonnet-4-5-20250929",
        "situational_analysis_model": "claude-sonnet-4-5-20250929",
        "memory_retrieval_model": "claude-sonnet-4-5-20250929",
        "memory_formation_model": "claude-sonnet-4-5-20250929",
        "trigger_compression_model": "claude-sonnet-4-5-20250929",
        "think_action_model": "claude-sonnet-4-5-20250929",
        "speak_action_model": "claude-sonnet-4-5-20250929",
        "visual_action_model": "claude-sonnet-4-5-20250929",
        "fetch_url_action_model": "claude-sonnet-4-5-20250929",
        "evaluate_priorities_action_model": "claude-sonnet-4-5-20250929",
        "tts_rewrite_model": "mistral-small3.2:latest",
    }

    try:
        with TestClient(production_app) as client:
            response = client.post("/api/model-config", json=payload)
    finally:
        Config._model_config = original_model_config
        Config._model_config_path = original_model_config_path
        production_app.state.agent_manager = original_agent_manager
        production_app.state.agent_startup_error = original_startup_error

    assert response.status_code == 400
    assert "not a known Anthropic model" in response.json()["detail"]


def test_pull_and_delete_ollama_model_routes_delegate_to_client():
    ollama_client = DummyOllamaClient()
    original_model_config = Config._model_config
    Config._model_config = ModelConfig(
        state_initialization_model=SupportedModel.CLAUDE_SONNET_4_5,
        action_planning_model=SupportedModel.CLAUDE_SONNET_4_5,
        situational_analysis_model=SupportedModel.CLAUDE_SONNET_4_5,
        memory_retrieval_model=SupportedModel.CLAUDE_SONNET_4_5,
        memory_formation_model=SupportedModel.CLAUDE_SONNET_4_5,
        trigger_compression_model=SupportedModel.CLAUDE_SONNET_4_5,
        think_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        speak_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        visual_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        fetch_url_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        evaluate_priorities_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        tts_rewrite_model=SupportedModel.MISTRAL_SMALL_3_2_Q4,
    )
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: ollama_client,
    )
    app.router.routes.extend(production_app.router.routes)

    try:
        with TestClient(app) as client:
            pull_response = client.post(
                "/api/ollama/models/pull", json={"name": "mistral-small3.2:latest"}
            )
            delete_response = client.delete(
                "/api/ollama/models/mistral-small3.2:latest"
            )
    finally:
        Config._model_config = original_model_config

    assert pull_response.status_code == 200
    assert pull_response.json()["name"] == "mistral-small3.2:latest"
    assert "Pulled Ollama model" in pull_response.json()["message"]
    assert ollama_client.pulled == ["mistral-small3.2:latest"]

    assert delete_response.status_code == 200
    assert delete_response.json()["name"] == "mistral-small3.2:latest"
    assert "Deleted Ollama model" in delete_response.json()["message"]
    assert ollama_client.deleted == ["mistral-small3.2:latest"]


def test_delete_ollama_model_rejects_models_still_referenced_in_config():
    ollama_client = DummyOllamaClient()
    original_model_config = Config._model_config
    Config._model_config = ModelConfig(
        state_initialization_model=SupportedModel.MISTRAL_SMALL_3_2,
        action_planning_model=SupportedModel.CLAUDE_SONNET_4_5,
        situational_analysis_model=SupportedModel.CLAUDE_SONNET_4_5,
        memory_retrieval_model=SupportedModel.CLAUDE_SONNET_4_5,
        memory_formation_model=SupportedModel.CLAUDE_SONNET_4_5,
        trigger_compression_model=SupportedModel.CLAUDE_SONNET_4_5,
        think_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        speak_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        visual_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        fetch_url_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        evaluate_priorities_action_model=SupportedModel.CLAUDE_SONNET_4_5,
        tts_rewrite_model=SupportedModel.MISTRAL_SMALL_3_2_Q4,
    )
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: ollama_client,
    )
    app.router.routes.extend(production_app.router.routes)

    try:
        with TestClient(app) as client:
            response = client.delete("/api/ollama/models/mistral-small3.2:latest")
    finally:
        Config._model_config = original_model_config

    assert response.status_code == 409
    assert "state_initialization_model" in response.json()["detail"]
    assert ollama_client.deleted == []


def test_get_installed_ollama_models_returns_502_when_ollama_is_unavailable():
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: DummyOllamaClient(
            list_error=RuntimeError("connection refused")
        ),
    )
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.get("/api/ollama/models")

    assert response.status_code == 502
    assert response.json()["detail"] == "Failed to reach Ollama: connection refused"


def test_delete_ollama_model_preserves_upstream_ollama_status_codes():
    app = create_app(
        initialize_on_startup=False,
        ollama_client_factory=lambda: DummyOllamaClient(
            delete_error=ollama.ResponseError("model not found", status_code=404)
        ),
    )
    app.router.routes.extend(production_app.router.routes)

    with TestClient(app) as client:
        response = client.delete("/api/ollama/models/missing-model:latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "model not found"
