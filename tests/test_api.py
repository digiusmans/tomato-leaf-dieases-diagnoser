
from fastapi.testclient import TestClient
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to Tomato Disease Detection API"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("document.pdf", b"test content", "application/pdf")}
    )

    assert response.status_code == 400


def test_corrupted_image():
    response = client.post(
        "/predict",
        files={"file": ("corrupted.jpg", b"invalid image data", "image/jpeg")}
    )

    assert response.status_code == 400


def test_file_size_limit():
    large_file = b"x" * (5 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_file, "image/jpeg")}
    )

    assert response.status_code == 413
