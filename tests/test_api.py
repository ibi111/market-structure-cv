from fastapi.testclient import TestClient
from api.main import app
import numpy as np
import cv2

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] == True


def test_info():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "FVG Detector"
    assert "bullish_fvg" in data["classes"]
    assert "bearish_fvg" in data["classes"]


def test_predict_valid_image():
    # Create a dummy white image
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.png', img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "total_detections" in data
    assert "bullish_count" in data
    assert "bearish_count" in data
    assert "inference_time_ms" in data


def test_predict_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_corrupted_image():
    response = client.post(
        "/predict",
        files={"file": ("test.png", b"corrupted_data", "image/png")}
    )
    assert response.status_code == 400