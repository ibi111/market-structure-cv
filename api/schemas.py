from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    cls: str
    confidence: float
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    detections: List[Detection]
    total_detections: int
    bullish_count: int
    bearish_count: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    name: str
    version: str
    classes: List[str]
    description: str