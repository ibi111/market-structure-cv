import cv2
import numpy as np
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.requests import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from schemas import (
    PredictionResponse,
    Detection,
    BoundingBox,
    HealthResponse,
    InfoResponse
)

# Router
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/", response_model=InfoResponse)
def info():
    return InfoResponse(
        name="FVG Detector",
        version="1.0.0",
        classes=["bullish_fvg", "bearish_fvg"],
        description="YOLO-based Fair Value Gap detector for NQ futures charts."
    )


@router.get("/health", response_model=HealthResponse)
def health():
    from main import model
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    from main import model, config

    # Validate file type
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PNG and JPEG are supported."
        )

    # Read and decode image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image."
        )

    # Run inference
    start = time.perf_counter()
    results = model(frame, conf=config['model']['conf_threshold'], verbose=False)[0]
    inference_time_ms = (time.perf_counter() - start) * 1000

    # Parse detections
    detections = []
    bullish_count = 0
    bearish_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = round(float(box.conf[0]), 4)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == 'bullish_fvg':
            bullish_count += 1
        else:
            bearish_count += 1

        detections.append(Detection(
            cls=cls_name,
            confidence=conf,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        ))

    return PredictionResponse(
        detections=detections,
        total_detections=len(detections),
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        inference_time_ms=round(inference_time_ms, 2)
    )