from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.concurrency import run_in_threadpool
from pixeldebug.limits import UploadLimitMiddleware
from pixeldebug.vision import compare, decode_image


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIXELDEBUG_", env_file=".env", extra="ignore"
    )
    max_upload_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 12_000_000
    allowed_origins: list[str] = ["http://localhost:5173"]
    max_concurrent_analyses: int = 2


class Box(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Defect(BaseModel):
    id: int
    bbox: Box
    category: str
    source: Literal["cv_heuristic"]
    changed_pixels: int
    explanation: str


class Analysis(BaseModel):
    width: int
    height: int
    ssim: float
    changed_fraction: float
    alignment: dict
    defects: list[Defect]
    overlay: str
    diagnosis_source: Literal["deterministic_cv"]
    warnings: list[str]


settings = Settings()
app = FastAPI(title="PixelDebug", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.add_middleware(UploadLimitMiddleware, settings=settings)
slots = asyncio.Semaphore(settings.max_concurrent_analyses)
logger = logging.getLogger("pixeldebug")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.middleware("http")
async def request_log(request: Request, call_next):
    started = time.monotonic()
    request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
            }
        )
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/model/info")
def model_info():
    return {
        "cv": "ready",
        "multimodal": "experimental_offline_only",
        "pretrained_validated": False,
        "architecture": "SigLIP2 patch tokens → trainable MLP → Gemma causal LM",
        "reason": "A task-trained, evaluated checkpoint is required before serving AI diagnoses.",
    }


async def read_upload(upload: UploadFile) -> bytes:
    try:
        data = await upload.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                413,
                f"Each screenshot must be at most {settings.max_upload_bytes} bytes",
            )
        if not data:
            raise HTTPException(422, "Screenshot is empty")
        return data
    finally:
        await upload.close()


@app.post("/analyze", response_model=Analysis)
async def analyze(
    reference: UploadFile = File(...),
    defective: UploadFile = File(...),
    sensitivity: float = Form(0.5, ge=0, le=1),
    align: bool = Form(False),
):
    a, b = await read_upload(reference), await read_upload(defective)

    def process():
        return compare(
            decode_image(a, settings.max_pixels),
            decode_image(b, settings.max_pixels),
            sensitivity,
            align=align,
        )

    try:
        async with slots:
            return await run_in_threadpool(process)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/predict")
def predict():
    raise HTTPException(
        503,
        "Multimodal serving is unavailable: train and evaluate a checkpoint with pixeldebug.ml first. Use /analyze for working CV diagnostics.",
    )
