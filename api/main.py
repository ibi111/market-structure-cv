import yaml
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
ENV = os.getenv("ENV", "development")

# Load config
def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Load model
model_path = Path(__file__).parent.parent / config['model']['path']
model = YOLO(str(model_path))

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# App
app = FastAPI(
    title="FVG Detector API",
    description="Detects Fair Value Gaps in NQ futures chart images.",
    version="1.0.0",
    docs_url=None if ENV == "production" else "/docs",
    redoc_url=None if ENV == "production" else "/redoc",
    openapi_url=None if ENV == "production" else "/openapi.json"
)

# Rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routes
from routes import router
app.include_router(router)