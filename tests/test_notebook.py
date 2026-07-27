from fastapi.testclient import TestClient
from ai_studio.server.app import app
from ai_studio.notebook.manager import notebook_manager

client = TestClient(app)

def test_create_and_get_note():
    note = notebook_manager.create_note("Test Research", "Deep learning content", ["ai", "test"])
    assert note["title"] == "Test Research"

    notes = notebook_manager.get_notes()
    assert len(notes) > 0
    assert any(n["title"] == "Test Research" for n in notes)

def test_notebook_api_endpoints():
    response = client.get("/v1/notebook/notes")
    assert response.status_code == 200
    data = response.json()
    assert "notes" in data
