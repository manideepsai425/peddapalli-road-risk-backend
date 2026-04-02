"""
model_loader.py
===============
Self-contained ML training + road network data.
Extracted from peddapalli_road_risk_colab_v5.py and adapted for FastAPI.

Usage:
    loader = ModelLoader()
    loader.train()          # call once on startup
    loader.predict_risk(...)
    loader.get_current_weather(time_pref)
"""

import math as _math
import os
import warnings
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# ── Encoding dicts (same as Colab script) ────────────────────
ROAD_RISK_MAP = {"highway": 3, "junction": 3, "rural road": 2, "urban road": 1}
ROAD_TYPE_ENC = {"highway": 3, "junction": 2, "rural road": 1, "urban road": 0}
WEATHER_ENC   = {"foggy": 3, "rainy": 2, "cloudy": 1, "clear": 0}
CAUSE_CAT_ENC = {"speed": 3, "junction": 2, "road_design": 1, "other": 0}

REAL_HOTSPOTS = {
    "Katnapalli Gate",
    "Shastrynagar Sultanabad",
    "Andugulapalli Junction",
    "Basanthnagar Bridge",
    "Rajiv Rahadari GDK",
}

# ── Haversine / road geometry ────────────────────────────────
_TORTUOSITY = {"highway": 1.22, "junction": 1.15, "urban road": 1.12, "rural road": 1.38}
_SPEED_KMH  = {"highway": 60,   "junction": 25,   "urban road": 28,   "rural road": 38}


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = _math.radians(lat2 - lat1)
    dlon = _math.radians(lon2 - lon1)
    a = (_math.sin(dlat / 2) ** 2 +
         _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2)) *
         _math.sin(dlon / 2) ** 2)
    return R * 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1 - a))


# ── Minimal fallback CSV (used when data/accidents.csv is absent) ─
_STUB_CSV = """year,mandal,accident_prone_area,latitude,longitude,road_type,vehicles,fatalities,injuries,accident_type,weather,cause
2020,Peddapalli,Peddapalli Bus Stand,18.6155,79.3744,junction,2,1,2,bike-auto,collision,clear junction design issue
2020,Peddapalli,Andugulapalli Junction,18.6360,79.3920,junction,3,1,4,truck-car,collision,clear overspeeding
2020,Ramagundam,Godavarikhani Chowrastha,18.7502,79.4821,junction,2,1,2,truck-car,collision,clear junction design issue
2020,Ramagundam,Rajiv Rahadari GDK,18.7410,79.4580,highway,3,2,3,truck-car,collision,clear overspeeding
2020,Sultanabad,Katnapalli Gate,18.5610,79.3520,highway,3,1,4,truck-car,collision,foggy poor visibility
2020,Sultanabad,Shastrynagar Sultanabad,18.5490,79.3350,highway,2,1,2,head-on,collision,clear overspeeding
2020,Palakurthy,Basanthnagar Bridge,18.6810,79.4525,highway,2,2,1,head-on,collision,clear road engineering defect
2020,Manthani,Godavari Bridge Manthani,18.6625,79.6633,highway,2,1,2,car-lorry,collision,foggy road engineering defect
2020,Ramagiri,Kalvacharla Curve,18.6310,79.5410,highway,2,2,1,head-on,collision,clear overspeeding
2021,Peddapalli,Peddapalli Bus Stand,18.6155,79.3744,junction,2,0,2,bike-auto,collision,clear junction design issue
2021,Peddapalli,Andugulapalli Junction,18.6360,79.3920,junction,2,1,2,car-lorry,collision,clear overspeeding
2021,Ramagundam,Godavarikhani Chowrastha,18.7502,79.4821,junction,3,1,3,truck-auto,collision,clear junction design issue
2021,Ramagundam,Rajiv Rahadari GDK,18.7410,79.4580,highway,2,1,2,head-on,collision,clear overspeeding
2021,Sultanabad,Katnapalli Gate,18.5610,79.3520,highway,2,1,2,bike-lorry,collision,foggy poor visibility
2021,Sultanabad,Shastrynagar Sultanabad,18.5490,79.3350,highway,3,1,3,truck-car,collision,clear overspeeding
2021,Palakurthy,Basanthnagar Bridge,18.6810,79.4525,highway,2,2,1,head-on,collision,clear road engineering defect
2021,Manthani,Godavari Bridge Manthani,18.6625,79.6633,highway,2,1,2,car-lorry,collision,clear overspeeding
2021,Ramagiri,Kalvacharla Curve,18.6310,79.5410,highway,3,2,4,truck-car,collision,clear overspeeding
2022,Peddapalli,Peddapalli Bus Stand,18.6155,79.3744,junction,3,1,2,truck-car,collision,clear junction design issue
2022,Ramagundam,Godavarikhani Chowrastha,18.7502,79.4821,junction,2,0,4,auto-bike,collision,clear junction design issue
2022,Sultanabad,Katnapalli Gate,18.5610,79.3520,highway,2,2,1,head-on,collision,clear overspeeding
2022,Palakurthy,Basanthnagar Bridge,18.6810,79.4525,highway,2,1,3,head-on,collision,clear road engineering defect
2022,Manthani,Godavari Bridge Manthani,18.6625,79.6633,highway,1,1,0,truck-skid,foggy,poor visibility
2023,Peddapalli,Andugulapalli Junction,18.6360,79.3920,junction,2,1,2,car-lorry,collision,clear overspeeding
2023,Ramagundam,Rajiv Rahadari GDK,18.7410,79.4580,highway,3,2,3,truck-car,collision,clear overspeeding
2023,Sultanabad,Shastrynagar Sultanabad,18.5490,79.3350,highway,2,1,2,car-lorry,collision,clear overspeeding
2024,Peddapalli,Peddapalli Bus Stand,18.6155,79.3744,junction,2,1,3,lorry-car,collision,cloudy junction design issue
2024,Ramagundam,Godavarikhani Chowrastha,18.7502,79.4821,junction,1,1,0,pedestrian,hit,clear poor lighting
2024,Sultanabad,Katnapalli Gate,18.5610,79.3520,highway,3,1,3,truck-car,collision,clear overspeeding
2024,Palakurthy,Basanthnagar Bridge,18.6810,79.4525,highway,2,2,1,head-on,collision,foggy road engineering defect
2025,Peddapalli,Andugulapalli Junction,18.6360,79.3920,junction,2,1,2,car-lorry,collision,cloudy junction design issue
2025,Ramagundam,Rajiv Rahadari GDK,18.7410,79.4580,highway,2,1,1,rear-end,collision,clear overspeeding
2025,Sultanabad,Shastrynagar Sultanabad,18.5490,79.3350,highway,2,2,1,car-lorry,collision,clear overspeeding
2025,Manthani,Godavari Bridge Manthani,18.6625,79.6633,highway,2,1,2,car-lorry,collision,clear overspeeding
"""


class ModelLoader:
    """Encapsulates training state and inference. One instance per process."""

    # ── GPS-Accurate Segment Data (68 nodes) ─────────────────
    SEGMENT_DATA = {
        "Peddapalli Bus Stand":        {"mandal": "Peddapalli",  "lat": 18.6155, "lon": 79.3744, "road_type": "junction"},
        "Peddapalli Railway Station":  {"mandal": "Peddapalli",  "lat": 18.6189, "lon": 79.3812, "road_type": "urban road"},
        "Andugulapalli Junction":      {"mandal": "Peddapalli",  "lat": 18.6360, "lon": 79.3920, "road_type": "junction"},
        "Appannapeta Bridge":          {"mandal": "Peddapalli",  "lat": 18.6341, "lon": 79.3950, "road_type": "highway"},
        "Peddakalvala Bypass":         {"mandal": "Peddapalli",  "lat": 18.6455, "lon": 79.3988, "road_type": "highway"},
        "Rangampalli Cross Road":      {"mandal": "Peddapalli",  "lat": 18.6121, "lon": 79.3785, "road_type": "junction"},
        "Sabbitham Road Curve":        {"mandal": "Peddapalli",  "lat": 18.5902, "lon": 79.3455, "road_type": "rural road"},
        "Godavari Road Peddapalli":    {"mandal": "Peddapalli",  "lat": 18.6220, "lon": 79.3870, "road_type": "urban road"},
        "Sultanabad Town":             {"mandal": "Sultanabad",  "lat": 18.5285, "lon": 79.3195, "road_type": "junction"},
        "Katnapalli Gate":             {"mandal": "Sultanabad",  "lat": 18.5610, "lon": 79.3520, "road_type": "highway"},
        "Shastrynagar Sultanabad":     {"mandal": "Sultanabad",  "lat": 18.5490, "lon": 79.3350, "road_type": "highway"},
        "Poosala Curve":               {"mandal": "Sultanabad",  "lat": 18.5450, "lon": 79.3360, "road_type": "highway"},
        "Neerukulla Bridge":           {"mandal": "Sultanabad",  "lat": 18.5720, "lon": 79.3650, "road_type": "highway"},
        "Kodurupaka Curve":            {"mandal": "Sultanabad",  "lat": 18.5850, "lon": 79.3780, "road_type": "rural road"},
        "Srirampur X Road":            {"mandal": "Srirampur",   "lat": 18.5255, "lon": 79.3142, "road_type": "junction"},
        "Rajiv Rahadari Srirampur":    {"mandal": "Srirampur",   "lat": 18.5210, "lon": 79.3110, "road_type": "highway"},
        "Nagepalli Curve":             {"mandal": "Srirampur",   "lat": 18.5340, "lon": 79.3250, "road_type": "highway"},
        "Basanthnagar Bridge":         {"mandal": "Palakurthy",  "lat": 18.6810, "lon": 79.4525, "road_type": "highway"},
        "Palakurthy X Road":           {"mandal": "Palakurthy",  "lat": 18.6655, "lon": 79.4340, "road_type": "junction"},
        "Godavarikhani Link Road":     {"mandal": "Palakurthy",  "lat": 18.7020, "lon": 79.4650, "road_type": "highway"},
        "Kukkalagudur Stage":          {"mandal": "Palakurthy",  "lat": 18.6510, "lon": 79.4260, "road_type": "junction"},
        "Jakkapur Curve":              {"mandal": "Palakurthy",  "lat": 18.6250, "lon": 79.4050, "road_type": "rural road"},
        "Basanthnagar Junction":       {"mandal": "Anthergaon",  "lat": 18.6810, "lon": 79.4125, "road_type": "junction"},
        "Basanthnagar X Road":         {"mandal": "Anthergaon",  "lat": 18.7120, "lon": 79.4180, "road_type": "junction"},
        "Anthergaon Railway Gate":     {"mandal": "Anthergaon",  "lat": 18.7310, "lon": 79.4350, "road_type": "rural road"},
        "Golliwada Curve":             {"mandal": "Anthergaon",  "lat": 18.7450, "lon": 79.4520, "road_type": "rural road"},
        "Eklaspur Stage":              {"mandal": "Anthergaon",  "lat": 18.7210, "lon": 79.3920, "road_type": "rural road"},
        "Godavarikhani Chowrastha":    {"mandal": "Ramagundam",  "lat": 18.7502, "lon": 79.4821, "road_type": "junction"},
        "Rajiv Rahadari GDK":          {"mandal": "Ramagundam",  "lat": 18.7410, "lon": 79.4580, "road_type": "highway"},
        "Godavarikhani Bridge":        {"mandal": "Ramagundam",  "lat": 18.7856, "lon": 79.4812, "road_type": "highway"},
        "Medipally Cross Road":        {"mandal": "Ramagundam",  "lat": 18.7152, "lon": 79.4623, "road_type": "highway"},
        "NTPC TTS Main Gate":          {"mandal": "Ramagundam",  "lat": 18.7488, "lon": 79.5055, "road_type": "junction"},
        "Bypass Road Junction":        {"mandal": "Ramagundam",  "lat": 18.7712, "lon": 79.4621, "road_type": "junction"},
        "Old Bus Stand GDK":           {"mandal": "Ramagundam",  "lat": 18.7481, "lon": 79.4833, "road_type": "junction"},
        "Janagaon Cross Roads":        {"mandal": "Ramagundam",  "lat": 18.7121, "lon": 79.4325, "road_type": "highway"},
        "Malkapur NH Stretch":         {"mandal": "Ramagundam",  "lat": 18.7845, "lon": 79.5123, "road_type": "highway"},
        "Ramagundam Railway Station":  {"mandal": "Ramagundam",  "lat": 18.7589, "lon": 79.4456, "road_type": "urban road"},
        "FCI Road Entrance":           {"mandal": "Ramagundam",  "lat": 18.7612, "lon": 79.4910, "road_type": "rural road"},
        "Fertilizer City Gate":        {"mandal": "Ramagundam",  "lat": 18.7698, "lon": 79.5022, "road_type": "junction"},
        "Subhash Nagar NH":            {"mandal": "Ramagundam",  "lat": 18.7755, "lon": 79.5055, "road_type": "highway"},
        "Yellampalli Bridge Approach": {"mandal": "Ramagundam",  "lat": 18.8102, "lon": 79.3951, "road_type": "highway"},
        "Manthani Bus Stand":          {"mandal": "Manthani",    "lat": 18.6555, "lon": 79.6712, "road_type": "junction"},
        "Godavari Bridge Manthani":    {"mandal": "Manthani",    "lat": 18.6625, "lon": 79.6633, "road_type": "highway"},
        "Kannala Cross Road":          {"mandal": "Manthani",    "lat": 18.6144, "lon": 79.6211, "road_type": "junction"},
        "Peddapalli Manthani Hwy":     {"mandal": "Manthani",    "lat": 18.6422, "lon": 79.6122, "road_type": "highway"},
        "Kamanpur Cross Road":         {"mandal": "Kamanpur",    "lat": 18.6712, "lon": 79.5955, "road_type": "junction"},
        "Sundilla Reservoir Rd":       {"mandal": "Kamanpur",    "lat": 18.7210, "lon": 79.6450, "road_type": "rural road"},
        "Kaman Centre Town":           {"mandal": "Kamanpur",    "lat": 18.6655, "lon": 79.5880, "road_type": "urban road"},
        "Ramagiri Fort Road":          {"mandal": "Ramagiri",    "lat": 18.5830, "lon": 79.4920, "road_type": "rural road"},
        "Centenary Colony X Road":     {"mandal": "Ramagiri",    "lat": 18.6120, "lon": 79.5250, "road_type": "junction"},
        "Kalvacharla Curve":           {"mandal": "Ramagiri",    "lat": 18.6310, "lon": 79.5410, "road_type": "highway"},
        "Pannur Crossroad":            {"mandal": "Ramagiri",    "lat": 18.5520, "lon": 79.4620, "road_type": "junction"},
        "Dharmaram X Road":            {"mandal": "Dharmaram",   "lat": 18.6942, "lon": 79.2550, "road_type": "junction"},
        "Dongatuniki Gate":            {"mandal": "Dharmaram",   "lat": 18.7215, "lon": 79.2885, "road_type": "highway"},
        "Mallapur Curve":              {"mandal": "Dharmaram",   "lat": 18.6720, "lon": 79.2310, "road_type": "rural road"},
        "Pathapally Bridge":           {"mandal": "Dharmaram",   "lat": 18.6585, "lon": 79.2150, "road_type": "highway"},
        "Eligaid X Road":              {"mandal": "Eligaid",     "lat": 18.5280, "lon": 79.2810, "road_type": "junction"},
        "Muppirithota Junction":       {"mandal": "Eligaid",     "lat": 18.5360, "lon": 79.3020, "road_type": "junction"},
        "Sultanabad Eligaid Link":     {"mandal": "Eligaid",     "lat": 18.5120, "lon": 79.3150, "road_type": "rural road"},
        "Julapalli X Road":            {"mandal": "Julapalli",   "lat": 18.6015, "lon": 79.3240, "road_type": "junction"},
        "Peddapalli Julapalli Link":   {"mandal": "Julapalli",   "lat": 18.6050, "lon": 79.3580, "road_type": "rural road"},
        "Telukunta Curve":             {"mandal": "Julapalli",   "lat": 18.5820, "lon": 79.3455, "road_type": "rural road"},
        "Odela X Road":                {"mandal": "Odela",       "lat": 18.4912, "lon": 79.4125, "road_type": "junction"},
        "Odela Railway Gate":          {"mandal": "Odela",       "lat": 18.4855, "lon": 79.4080, "road_type": "rural road"},
        "Gumpula Stage":               {"mandal": "Odela",       "lat": 18.5320, "lon": 79.4610, "road_type": "junction"},
        "Parupalli X Road":            {"mandal": "Mutharam",    "lat": 18.4710, "lon": 79.5320, "road_type": "junction"},
        "Manthani Mutharam Road":      {"mandal": "Mutharam",    "lat": 18.4650, "lon": 79.5250, "road_type": "rural road"},
        "Odedu Bridge Approach":       {"mandal": "Mutharam",    "lat": 18.4950, "lon": 79.5580, "road_type": "rural road"},
    }

    LOCATION_ALIASES = {
        "peddapalli":           "Peddapalli Bus Stand",
        "peddapalli town":      "Peddapalli Bus Stand",
        "peddapalli bus stand": "Peddapalli Bus Stand",
        "peddapalli railway":   "Peddapalli Railway Station",
        "ramagundam":           "Old Bus Stand GDK",
        "godavarikhani":        "Godavarikhani Chowrastha",
        "gdkh":                 "Godavarikhani Chowrastha",
        "gdk":                  "Old Bus Stand GDK",
        "manthani":             "Manthani Bus Stand",
        "sultanabad":           "Sultanabad Town",
        "kamanpur":             "Kamanpur Cross Road",
        "ramagiri":             "Centenary Colony X Road",
        "dharmaram":            "Dharmaram X Road",
        "eligaid":              "Eligaid X Road",
        "julapalli":            "Julapalli X Road",
        "palakurthy":           "Palakurthy X Road",
        "odela":                "Odela X Road",
        "mutharam":             "Parupalli X Road",
        "srirampur":            "Srirampur X Road",
        "anthergaon":           "Basanthnagar X Road",
        "basanthnagar":         "Basanthnagar Junction",
        "ntpc":                 "NTPC TTS Main Gate",
        "yellampalli":          "Yellampalli Bridge Approach",
        "katnapalli":           "Katnapalli Gate",
    }

    _EDGE_DEFS = [
        ("Peddapalli Bus Stand",       "Rangampalli Cross Road",      "highway"),
        ("Peddapalli Bus Stand",       "Andugulapalli Junction",      "highway"),
        ("Peddapalli Bus Stand",       "Peddapalli Railway Station",  "urban road"),
        ("Peddapalli Bus Stand",       "Julapalli X Road",            "rural road"),
        ("Peddapalli Bus Stand",       "Sabbitham Road Curve",        "rural road"),
        ("Peddapalli Bus Stand",       "Sultanabad Town",             "highway"),
        ("Peddapalli Bus Stand",       "Pathapally Bridge",           "highway"),
        ("Peddapalli Bus Stand",       "Dharmaram X Road",            "highway"),
        ("Peddapalli Bus Stand",       "Peddapalli Manthani Hwy",     "highway"),
        ("Peddapalli Bus Stand",       "Eklaspur Stage",              "rural road"),
        ("Peddapalli Bus Stand",       "Odela X Road",                "rural road"),
        ("Rangampalli Cross Road",     "Andugulapalli Junction",      "highway"),
        ("Andugulapalli Junction",     "Appannapeta Bridge",          "highway"),
        ("Appannapeta Bridge",         "Peddakalvala Bypass",         "highway"),
        ("Peddakalvala Bypass",        "Basanthnagar Junction",       "highway"),
        ("Basanthnagar Junction",      "Godavarikhani Chowrastha",    "highway"),
        ("Basanthnagar Junction",      "Basanthnagar X Road",         "junction"),
        ("Basanthnagar Junction",      "Palakurthy X Road",           "highway"),
        ("Sultanabad Town",            "Shastrynagar Sultanabad",     "highway"),
        ("Shastrynagar Sultanabad",    "Katnapalli Gate",             "highway"),
        ("Katnapalli Gate",            "Poosala Curve",               "highway"),
        ("Katnapalli Gate",            "Neerukulla Bridge",           "highway"),
        ("Neerukulla Bridge",          "Kodurupaka Curve",            "rural road"),
        ("Sultanabad Town",            "Srirampur X Road",            "junction"),
        ("Srirampur X Road",           "Rajiv Rahadari Srirampur",    "highway"),
        ("Rajiv Rahadari Srirampur",   "Nagepalli Curve",             "highway"),
        ("Sultanabad Town",            "Eligaid X Road",              "rural road"),
        ("Eligaid X Road",             "Muppirithota Junction",       "junction"),
        ("Eligaid X Road",             "Sultanabad Eligaid Link",     "rural road"),
        ("Sultanabad Eligaid Link",    "Julapalli X Road",            "rural road"),
        ("Palakurthy X Road",          "Kukkalagudur Stage",          "junction"),
        ("Palakurthy X Road",          "Jakkapur Curve",              "rural road"),
        ("Palakurthy X Road",          "Godavarikhani Link Road",     "highway"),
        ("Godavarikhani Link Road",    "Godavarikhani Chowrastha",    "highway"),
        ("Basanthnagar Bridge",        "Palakurthy X Road",           "highway"),
        ("Basanthnagar Bridge",        "Basanthnagar Junction",       "highway"),
        ("Godavarikhani Chowrastha",   "Rajiv Rahadari GDK",          "highway"),
        ("Godavarikhani Chowrastha",   "Old Bus Stand GDK",           "junction"),
        ("Godavarikhani Chowrastha",   "Medipally Cross Road",        "highway"),
        ("Godavarikhani Chowrastha",   "Janagaon Cross Roads",        "highway"),
        ("Godavarikhani Chowrastha",   "Basanthnagar X Road",         "rural road"),
        ("Godavarikhani Chowrastha",   "Godavarikhani Bridge",        "highway"),
        ("Godavarikhani Chowrastha",   "Centenary Colony X Road",     "highway"),
        ("Rajiv Rahadari GDK",         "Medipally Cross Road",        "highway"),
        ("Medipally Cross Road",       "Janagaon Cross Roads",        "highway"),
        ("Janagaon Cross Roads",       "Ramagundam Railway Station",  "urban road"),
        ("Ramagundam Railway Station", "Old Bus Stand GDK",           "urban road"),
        ("Old Bus Stand GDK",          "Bypass Road Junction",        "junction"),
        ("Bypass Road Junction",       "Fertilizer City Gate",        "junction"),
        ("Fertilizer City Gate",       "NTPC TTS Main Gate",          "junction"),
        ("NTPC TTS Main Gate",         "Malkapur NH Stretch",         "highway"),
        ("Malkapur NH Stretch",        "Subhash Nagar NH",            "highway"),
        ("Godavarikhani Bridge",       "Yellampalli Bridge Approach", "highway"),
        ("FCI Road Entrance",          "Fertilizer City Gate",        "rural road"),
        ("Subhash Nagar NH",           "FCI Road Entrance",           "highway"),
        ("Basanthnagar X Road",        "Anthergaon Railway Gate",     "rural road"),
        ("Anthergaon Railway Gate",    "Golliwada Curve",             "rural road"),
        ("Golliwada Curve",            "Godavarikhani Chowrastha",    "rural road"),
        ("Basanthnagar X Road",        "Eklaspur Stage",              "rural road"),
        ("Peddapalli Manthani Hwy",    "Kannala Cross Road",          "highway"),
        ("Kannala Cross Road",         "Manthani Bus Stand",          "junction"),
        ("Manthani Bus Stand",         "Godavari Bridge Manthani",    "highway"),
        ("Godavari Bridge Manthani",   "Kamanpur Cross Road",         "highway"),
        ("Kamanpur Cross Road",        "Kaman Centre Town",           "urban road"),
        ("Kamanpur Cross Road",        "Sundilla Reservoir Rd",       "rural road"),
        ("Kamanpur Cross Road",        "Godavarikhani Chowrastha",    "highway"),
        ("Godavari Road Peddapalli",   "Peddapalli Manthani Hwy",     "highway"),
        ("Godavari Road Peddapalli",   "Peddapalli Bus Stand",        "urban road"),
        ("Centenary Colony X Road",    "Kalvacharla Curve",           "highway"),
        ("Centenary Colony X Road",    "Ramagiri Fort Road",          "rural road"),
        ("Pannur Crossroad",           "Ramagiri Fort Road",          "rural road"),
        ("Pannur Crossroad",           "Odela X Road",                "rural road"),
        ("Dharmaram X Road",           "Dongatuniki Gate",            "highway"),
        ("Dongatuniki Gate",           "Basanthnagar Junction",       "highway"),
        ("Dharmaram X Road",           "Mallapur Curve",              "rural road"),
        ("Dharmaram X Road",           "Pathapally Bridge",           "highway"),
        ("Pathapally Bridge",          "Peddapalli Bus Stand",        "highway"),
        ("Odela X Road",               "Odela Railway Gate",          "rural road"),
        ("Odela X Road",               "Gumpula Stage",               "junction"),
        ("Gumpula Stage",              "Pannur Crossroad",            "rural road"),
        ("Odela X Road",               "Parupalli X Road",            "rural road"),
        ("Parupalli X Road",           "Manthani Mutharam Road",      "rural road"),
        ("Parupalli X Road",           "Odedu Bridge Approach",       "rural road"),
        ("Odedu Bridge Approach",      "Manthani Bus Stand",          "rural road"),
        ("Manthani Mutharam Road",     "Manthani Bus Stand",          "rural road"),
        ("Julapalli X Road",           "Peddapalli Julapalli Link",   "rural road"),
        ("Julapalli X Road",           "Telukunta Curve",             "rural road"),
        ("Peddapalli Julapalli Link",  "Palakurthy X Road",           "rural road"),
        ("Telukunta Curve",            "Sultanabad Town",             "rural road"),
    ]

    def __init__(self):
        self._ready        = False
        self._model        = None
        self._scaler       = None
        self._best_name    = None
        self._mandal_sev   = {}
        self._hist_cnt_map = {}
        self._results      = {}

        # Pre-compute edge list
        self.EDGE_LIST = []
        for a, b, rt in self._EDGE_DEFS:
            if a in self.SEGMENT_DATA and b in self.SEGMENT_DATA:
                km = self._road_km(a, b, rt)
                tm = self._mins_travel(km, rt)
                self.EDGE_LIST.append((a, b, km, tm, rt))

    # ── Public interface ─────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return self._best_name or "Unknown"

    def get_model_accuracy(self) -> float:
        return self._results.get(self._best_name, {}).get("accuracy", 0.0)

    def resolve_location(self, name: str):
        key = name.lower().strip()
        if key in self.LOCATION_ALIASES:
            return self.LOCATION_ALIASES[key]
        for seg in self.SEGMENT_DATA:
            if seg.lower() == key:
                return seg
        for seg in self.SEGMENT_DATA:
            if key in seg.lower() or seg.lower() in key:
                return seg
        return None

    def train(self):
        import logging
        log = logging.getLogger(__name__)
        log.info("🚀 Training ML model ...")

        df = self._load_data()
        df_ml, feature_cols = self._engineer_features(df)

        X = df_ml[feature_cols].values
        y = df_ml["high_risk"].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # SMOTE if imbalanced
        ratio = y_tr.sum() / len(y_tr)
        if ratio < 0.4 or ratio > 0.6:
            k = min(5, int(y_tr.sum()) - 1)
            if k >= 1:
                X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Random Forest — fixed hyperparameters for fast startup
        best_rf = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        best_rf.fit(X_tr, y_tr)

        # Gradient Boosting — fixed hyperparameters for fast startup
        best_gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        best_gb.fit(X_tr, y_tr)

        # Voting Ensemble
        voting = VotingClassifier(
            estimators=[("rf", best_rf), ("gb", best_gb)], voting="soft"
        )
        voting.fit(X_tr, y_tr)

        results = {}
        for name, mdl in [("Random Forest", best_rf), ("Gradient Boosting", best_gb), ("Voting Ensemble", voting)]:
            yp  = mdl.predict(X_te_s)
            ypr = mdl.predict_proba(X_te_s)[:, 1]
            results[name] = {
                "accuracy": accuracy_score(y_te, yp),
                "f1":       f1_score(y_te, yp, zero_division=0),
                "roc_auc":  roc_auc_score(y_te, ypr),
            }

        best_name  = max(results, key=lambda k: results[k]["accuracy"])
        model_map  = {"Random Forest": best_rf, "Gradient Boosting": best_gb, "Voting Ensemble": voting}
        best_model = model_map[best_name]

        # Store mandal severity & historical counts from original df
        df["severity_score"] = (df["fatalities"] * 3 + df["injuries"]) / max(df["fatalities"].max() * 3 + df["injuries"].max(), 1)
        self._mandal_sev   = df.groupby("mandal")["severity_score"].mean().to_dict()
        self._hist_cnt_map = df.groupby("accident_prone_area").size().to_dict()

        self._model     = best_model
        self._scaler    = scaler
        self._best_name = best_name
        self._results   = results
        self._ready     = True

        log.info(f"✅ Best model: {best_name} | Accuracy={results[best_name]['accuracy']:.4f}")

    def predict_risk(self, road_type: str, weather_cond: str, road_type_risk: int,
                     hist_count: int, mandal_rl: float,
                     is_rainy: int, is_foggy: int, is_cloudy: int,
                     multi_vehicle: int = 1, heavy_veh: int = 0,
                     year: int = 2025, vehicles: int = 2,
                     year_norm: float = 1.0, accident_intensity: float = 0.3,
                     weather_severity: int = None,
                     is_junction: int = 0, is_highway: int = 0,
                     segment_name: str = "") -> float:

        ws  = weather_severity or {"foggy": 4, "rainy": 3, "cloudy": 2, "clear": 1}.get(weather_cond, 1)
        rwi = road_type_risk * ws
        vsev = vehicles * accident_intensity
        cc   = CAUSE_CAT_ENC.get(
            "speed"       if road_type == "highway"    else
            "junction"    if road_type == "junction"   else
            "road_design" if road_type == "rural road" else "other", 0
        )

        row = np.array([[year, ROAD_TYPE_ENC.get(road_type, 2), vehicles,
                         WEATHER_ENC.get(weather_cond, 0), cc,
                         road_type_risk, hist_count, mandal_rl,
                         is_rainy, is_foggy, is_cloudy,
                         multi_vehicle, heavy_veh, year_norm,
                         accident_intensity, ws, rwi, vsev,
                         is_junction, is_highway]])

        n_feat = (self._model.estimators_[0].n_features_in_
                  if hasattr(self._model, "estimators_")
                  else self._model.n_features_in_)
        row    = row[:, :n_feat]
        row_sc = self._scaler.transform(row)

        prob  = float(self._model.predict_proba(row_sc)[0][1])
        mult  = {"rainy": 1.18, "foggy": 1.22, "cloudy": 1.08}.get(weather_cond, 1.0)
        score = float(np.clip(prob * mult, 0, 1))

        if segment_name in REAL_HOTSPOTS:
            score = max(score, 0.72)

        return score

    def get_current_weather(self, time_pref: str = "now") -> dict:
        cond, temp, rain = "clear", "N/A", 0
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": 18.61, "longitude": 79.37,
                        "current_weather": True, "hourly": "precipitation"},
                timeout=5
            )
            data = resp.json()
            cw   = data.get("current_weather", {})
            code = cw.get("weathercode", 0)
            temp = str(cw.get("temperature", "N/A"))
            cond = ("rainy"  if code in [51,53,55,61,63,65,80,81,82]
                    else "foggy"  if code in [71,73,75,77,85,86,45,48]
                    else "cloudy" if code in [1,2,3]
                    else "clear")
        except Exception:
            pass

        if time_pref and time_pref != "now":
            try:
                hour = int(time_pref.split(":")[0])
                if hour >= 20 or hour <= 5:
                    cond = "foggy"
            except Exception:
                pass

        return {"weather_condition": cond, "temperature_c": temp, "rainfall_mm": rain}

    # ── Private helpers ──────────────────────────────────────

    def _road_km(self, a: str, b: str, rt: str) -> float:
        sa, sb = self.SEGMENT_DATA[a], self.SEGMENT_DATA[b]
        raw    = _haversine(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
        return round(raw * _TORTUOSITY[rt], 2)

    def _mins_travel(self, km: float, rt: str) -> float:
        return round(km / _SPEED_KMH[rt] * 60, 1)

    def _load_data(self) -> pd.DataFrame:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "accidents.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.read_csv(StringIO(_STUB_CSV))

    def _engineer_features(self, df_raw: pd.DataFrame):
        df = df_raw.copy()

        def _exw(s):
            s = str(s).lower()
            for w in ["foggy", "rainy", "cloudy"]:
                if w in s: return w
            return "clear"

        def _exc(s):
            s = str(s).lower()
            for c in ["overspeeding","poor lighting","poor road condition",
                      "junction design issue","dangerous curve","lack of signage",
                      "road engineering defect","lane violation","poor visibility"]:
                if c in s: return c
            return "other"

        def _mc(c):
            c = str(c).lower()
            if "speed" in c: return "speed"
            if "junction" in c: return "junction"
            if "road" in c or "curve" in c or "engineering" in c: return "road_design"
            return "other"

        df["weather_condition"] = df["cause"].apply(_exw)
        df["cause_reason"]      = df["cause"].apply(_exc)
        df.rename(columns={"weather": "accident_movement"}, inplace=True)

        df["high_risk"] = ((df["fatalities"] > 0) | (df["injuries"] >= 2)).astype(int)

        mf, mi = max(df["fatalities"].max(), 1), max(df["injuries"].max(), 1)
        df["accident_intensity"]  = (df["fatalities"] / mf) * 0.6 + (df["injuries"] / mi) * 0.4

        wsev = {"foggy": 4, "rainy": 3, "cloudy": 2, "clear": 1}
        df["weather_severity"]         = df["weather_condition"].map(wsev).fillna(1)
        df["road_type_risk"]           = df["road_type"].map(ROAD_RISK_MAP).fillna(2).astype(int)
        df["road_weather_interaction"] = df["road_type_risk"] * df["weather_severity"]
        df["vehicle_severity"]         = df["vehicles"] * df["accident_intensity"]

        ms = max(df["fatalities"].max() * 3 + df["injuries"].max(), 1)
        df["severity_score"]      = (df["fatalities"] * 3 + df["injuries"]) / ms
        df["cause_category"]      = df["cause_reason"].apply(_mc)

        hist = df.groupby("accident_prone_area").size().to_dict()
        df["historical_accident_count"] = df["accident_prone_area"].map(hist)

        ms2 = df.groupby("mandal")["severity_score"].mean().to_dict()
        df["mandal_risk_level"] = df["mandal"].map(ms2)

        df["is_rainy"]  = (df["weather_condition"] == "rainy").astype(int)
        df["is_foggy"]  = (df["weather_condition"] == "foggy").astype(int)
        df["is_cloudy"] = (df["weather_condition"] == "cloudy").astype(int)
        df["multi_vehicle"]       = (df["vehicles"] > 1).astype(int)
        df["heavy_vehicle_proxy"] = df["accident_type"].str.contains(
            "truck|lorry|bus|tipper|dumper", case=False, na=False).astype(int)
        df["year_norm"]  = (df["year"] - df["year"].min()) / max(1, df["year"].max() - df["year"].min())
        df["is_junction"] = (df["road_type"] == "junction").astype(int)
        df["is_highway"]  = (df["road_type"] == "highway").astype(int)

        drop_cols = ["cause", "cause_reason", "accident_prone_area", "mandal",
                     "accident_type", "accident_movement", "latitude", "longitude",
                     "fatalities", "injuries", "severity_score"]
        df_ml = df.drop(columns=drop_cols, errors="ignore")

        for col in ["road_type", "weather_condition", "cause_category"]:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        feat_cols = [c for c in df_ml.columns if c != "high_risk"]
        return df_ml, feat_cols