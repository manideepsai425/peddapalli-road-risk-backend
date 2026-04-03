"""
main.py
=======
Peddapalli Road Risk — FastAPI Backend  v1.0

Startup:
    uvicorn main:app --host 0.0.0.0 --port 8000
    (Render will run: uvicorn main:app --host 0.0.0.0 --port $PORT)
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_loader import ModelLoader
from router import RoadRouter
from schemas import HeatmapResponse, RouteRequest, RouteResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

model_loader = ModelLoader()
road_router  = RoadRouter(model_loader)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 55)
    logger.info("  Peddapalli Road Risk API — starting up")
    logger.info("═" * 55)
    model_loader.train()
    logger.info(f"  Network: {len(model_loader.SEGMENT_DATA)} segments | {len(model_loader.EDGE_LIST)} edges")
    logger.info("  API ready ✅")
    yield


app = FastAPI(
    title="Peddapalli Road Risk API",
    description=(
        "ML-powered road risk prediction and Dijkstra-based routing "
        "for Peddapalli District, Telangana."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS env var on Render to your Vercel URL, e.g.:
# https://peddapalli-road-risk.vercel.app
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
_extra = [o.strip() for o in _raw_origins.split(",") if o.strip()]

_allow_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    *_extra,
]
# allow_credentials=True is incompatible with allow_origins=["*"]
# so use wildcard only when no specific origins are configured (dev mode)
_use_wildcard = not _extra

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _use_wildcard else _allow_origins,
    allow_credentials=False if _use_wildcard else True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/", tags=["Utility"])
def root_health():
    return {"status": "ok"}

@app.get("/health", tags=["Utility"])
def health_check():
    return {
        "status":       "ok" if model_loader.is_ready() else "training",
        "model_ready":  model_loader.is_ready(),
        "model_name":   model_loader.get_model_name(),
        "accuracy":     round(model_loader.get_model_accuracy(), 4),
        "segments":     len(model_loader.SEGMENT_DATA),
        "edges":        len(model_loader.EDGE_LIST),
    }


@app.get("/locations", tags=["Utility"])
def get_locations():
    return {
        "locations": [
            {
                "id":        name,
                "mandal":    info["mandal"],
                "lat":       info["lat"],
                "lon":       info["lon"],
                "road_type": info["road_type"],
            }
            for name, info in model_loader.SEGMENT_DATA.items()
        ],
        "aliases": list(model_loader.LOCATION_ALIASES.keys()),
    }


@app.post("/predict-route", response_model=RouteResponse, tags=["Prediction"])
def predict_route(req: RouteRequest):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model is still training. Please retry in a few seconds.")

    src = model_loader.resolve_location(req.origin)
    dst = model_loader.resolve_location(req.destination)

    if not src:
        raise HTTPException(
            status_code=400,
            detail=f"Origin '{req.origin}' not found. Use GET /locations for valid names."
        )
    if not dst:
        raise HTTPException(
            status_code=400,
            detail=f"Destination '{req.destination}' not found. Use GET /locations for valid names."
        )
    if src == dst:
        raise HTTPException(status_code=400, detail="Origin and destination must be different.")

    try:
        result = road_router.compute_routes(
            source=src,
            destination=dst,
            weather_condition=req.weather_condition,
            time_of_day=req.time_of_day or "now",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal routing error.")

    return result


@app.get("/heatmap", response_model=HeatmapResponse, tags=["Prediction"])
def get_heatmap(weather: str = "clear"):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    valid = {"clear", "rainy", "foggy", "cloudy"}
    if weather not in valid:
        raise HTTPException(status_code=400, detail=f"weather must be one of {sorted(valid)}")

    segments = road_router.get_heatmap(weather)
    return HeatmapResponse(segments=segments)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)