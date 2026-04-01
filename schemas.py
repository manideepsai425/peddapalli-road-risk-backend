"""
schemas.py — Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class RouteRequest(BaseModel):
    origin: str = Field(..., example="peddapalli")
    destination: str = Field(..., example="ramagundam")
    weather_condition: Optional[str] = Field(
        default=None,
        description="clear | rainy | foggy | cloudy. Omit to use live weather."
    )
    time_of_day: Optional[str] = Field(
        default="now",
        example="08:30",
        description="HH:MM or 'now'. After 20:00 auto-sets foggy."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "origin": "peddapalli",
                "destination": "ramagundam",
                "weather_condition": None,
                "time_of_day": "now"
            }
        }


class SegmentRisk(BaseModel):
    name: str
    lat: float
    lon: float
    risk_score: float
    risk_label: str
    risk_color: str
    road_type: str
    mandal: str
    factors: List[str]
    is_hotspot: bool
    weather: str


class RouteCoord(BaseModel):
    lat: float
    lon: float
    name: str
    risk_score: float
    risk_label: str
    risk_color: str
    is_hotspot: bool


class RouteVariant(BaseModel):
    variant: str
    label: str
    recommended: bool
    distance_km: float
    duration_min: int
    avg_risk_score: float
    risk_label: str
    risk_color: str
    path: List[str]
    coordinates: List[RouteCoord]
    hotspots: List[str]
    top_factors: List[str]


class WeatherInfo(BaseModel):
    condition: str
    temperature_c: str
    rainfall_mm: float


class RouteResponse(BaseModel):
    origin: str
    destination: str
    weather: WeatherInfo
    model_name: str
    model_accuracy: float
    routes: List[RouteVariant]
    recommendation: str
    top_risk_segments: List[SegmentRisk]


class HeatmapSegment(BaseModel):
    name: str
    lat: float
    lon: float
    risk_score: float
    risk_label: str
    risk_color: str
    road_type: str
    mandal: str
    is_hotspot: bool


class HeatmapResponse(BaseModel):
    segments: List[HeatmapSegment]