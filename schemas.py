"""
schemas.py — Pydantic models for request/response validation.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


VALID_WEATHER = Literal["clear", "rainy", "foggy", "cloudy"]


class RouteRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "origin": "peddapalli",
                "destination": "ramagundam",
                "weather_condition": None,
                "time_of_day": "now",
            }
        }
    )

    origin: str = Field(..., description="Origin location name or alias")
    destination: str = Field(..., description="Destination location name or alias")
    weather_condition: Optional[VALID_WEATHER] = Field(
        default=None,
        description="clear | rainy | foggy | cloudy. Omit to use live weather.",
    )
    time_of_day: Optional[str] = Field(
        default="now",
        description="HH:MM or 'now'. After 20:00 / before 06:00 auto-sets foggy.",
    )


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
