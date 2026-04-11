from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.api_server import app as production_app
from agent.api_server import create_app


class DummyManager:
    def __init__(self, name: str = "Chloe") -> None:
        self.state = SimpleNamespace(name=name)
        self.agent = SimpleNamespace(close=lambda: None)


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
