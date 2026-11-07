import io
from PIL import Image
from fastapi.testclient import TestClient
from pixeldebug.api import app, settings

client = TestClient(app)


def image(color="white", size=(40, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


def files(a=None, b=None):
    return {
        "reference": ("a.png", a if a is not None else image(), "image/png"),
        "defective": ("b.png", b if b is not None else image(), "image/png"),
    }


def test_health_and_model_truthfulness():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model/info").json()["pretrained_validated"] is False
    assert client.post("/predict").status_code == 503


def test_analysis():
    result = client.post("/analyze", files=files(b=image("black")))
    assert result.status_code == 200
    assert result.json()["defects"]
    assert result.headers["x-request-id"]


def test_invalid_inputs():
    assert client.post("/analyze", files=files(a=b"broken")).status_code == 422
    assert (
        client.post("/analyze", files=files(), data={"sensitivity": 2}).status_code
        == 422
    )
    assert (
        client.post("/analyze", files=files(b=image(size=(50, 50)))).status_code == 422
    )
    assert client.post("/analyze").status_code == 422


def test_file_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 20)
    assert client.post("/analyze", files=files()).status_code == 413


def test_cors():
    response = client.options(
        "/analyze",
        headers={
            "origin": "http://localhost:5173",
            "access-control-request-method": "POST",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_total_body_limit_with_and_without_content_length(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 20)
    assert client.post("/analyze", content=b"x" * 66000).status_code == 413

    def chunks():
        for _ in range(7):
            yield b"x" * 10000

    assert client.post("/analyze", content=chunks()).status_code == 413
    assert (
        client.post(
            "/analyze", content=b"", headers={"content-length": "nope"}
        ).status_code
        == 400
    )
