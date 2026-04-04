"""
router.py
=========
Dijkstra-based routing engine.
Builds the risk-weighted road graph and computes 3 route variants:
    safest   → minimise risk score
    fastest  → minimise travel time
    balanced → 50/50 blend of risk + time
"""

import heapq
from collections import defaultdict
from typing import Dict, List, Optional

from model_loader import ModelLoader, ROAD_RISK_MAP, REAL_HOTSPOTS
from schemas import (
    HeatmapSegment, RouteCoord, RouteResponse, RouteVariant,
    SegmentRisk, WeatherInfo,
)


def _risk_label(score: float) -> str:
    if score < 0.35: return "Low Risk"
    if score < 0.65: return "Medium Risk"
    return "High Risk"


def _risk_color(score: float) -> str:
    if score < 0.35: return "green"
    if score < 0.65: return "orange"
    return "red"


def _get_factors(road_type: str, weather: str, seg_name: str, hist: int) -> List[str]:
    factors = []
    if seg_name in REAL_HOTSPOTS:
        factors.append("⚠ CONFIRMED FATAL HOTSPOT (Telangana Today 2023-2025)")
    if road_type == "highway":
        factors.append("High-speed highway segment")
    if road_type == "junction":
        factors.append("Junction — elevated collision probability")
    if weather == "foggy":
        factors.append("⚠ Foggy / night conditions — poor visibility")
    if weather == "rainy":
        factors.append("Wet road surface — reduced traction")
    if weather == "cloudy":
        factors.append("Overcast — reduced visibility")
    if hist >= 5:
        factors.append(f"Accident-prone zone ({hist} recorded incidents)")
    elif hist >= 3:
        factors.append(f"Moderate accident history ({hist} incidents)")
    return factors


class RoadRouter:
    def __init__(self, loader: ModelLoader):
        self.loader = loader

    # ── Public API ───────────────────────────────────────────

    def compute_routes(
        self,
        source: str,
        destination: str,
        weather_condition: Optional[str],
        time_of_day: str,
    ) -> RouteResponse:

        # Resolve weather
        if weather_condition and weather_condition in {"clear", "rainy", "foggy", "cloudy"}:
            w_info = {"weather_condition": weather_condition, "temperature_c": "N/A", "rainfall_mm": 0}
        else:
            w_info = self.loader.get_current_weather(time_of_day)

        W  = w_info["weather_condition"]
        IR = int(W == "rainy")
        IF = int(W == "foggy")
        IC = int(W == "cloudy")

        graph, seg_risks = self._build_network(W, IR, IF, IC)

        # Three routing variants
        routes = []
        for key, weight in [("safest", "risk"), ("fastest", "time"), ("balanced", "balanced")]:
            path, trail = self._dijkstra(graph, source, destination, weight)
            if not path or len(path) < 2:
                continue

            km   = round(sum(e[2] for e in trail), 1)
            mins = int(round(sum(e[3] for e in trail), 0))
            avg  = round(sum(e[4] for e in trail) / len(trail), 4) if trail else 0.5
            hotspots = [s for s in path if seg_risks.get(s, {}).get("is_hotspot")]

            coords = [
                RouteCoord(
                    lat=seg_risks[s]["lat"],
                    lon=seg_risks[s]["lon"],
                    name=s,
                    risk_score=seg_risks[s]["risk_score"],
                    risk_label=seg_risks[s]["risk_label"],
                    risk_color=seg_risks[s]["risk_color"],
                    is_hotspot=seg_risks[s]["is_hotspot"],
                )
                for s in path if s in seg_risks
            ]

            # Deduplicated factors for this route
            seen, top_factors = set(), []
            for s in path:
                for f in seg_risks.get(s, {}).get("factors", []):
                    if f not in seen:
                        seen.add(f)
                        top_factors.append(f)
                        if len(top_factors) == 5:
                            break
                if len(top_factors) == 5:
                    break

            routes.append(RouteVariant(
                variant=key,
                label=f"{key.upper()} ROUTE",
                recommended=False,
                distance_km=km,
                duration_min=mins,
                avg_risk_score=avg,
                risk_label=_risk_label(avg),
                risk_color=_risk_color(avg),
                path=path,
                coordinates=coords,
                hotspots=hotspots,
                top_factors=top_factors,
            ))

        if not routes:
            raise ValueError("No route found between the selected locations.")

        routes.sort(key=lambda r: r.avg_risk_score)
        routes[0].recommended = True
        best = routes[0]

        worst_risk = max(r.avg_risk_score for r in routes)
        risk_cut   = round((1 - best.avg_risk_score / worst_risk) * 100) if worst_risk > 0 else 0
        recommendation = (
            f"Take the {best.label} — {risk_cut}% safer than the riskiest option. "
            f"{best.distance_km} km | {best.duration_min} min | {_risk_label(best.avg_risk_score)}"
        )

        # Top 5 risk segments across all routes
        all_segs = {s for r in routes for s in r.path}
        top5 = sorted(
            [(s, seg_risks[s]) for s in all_segs if s in seg_risks],
            key=lambda x: -x[1]["risk_score"]
        )[:5]

        top_risk_segs = [
            SegmentRisk(
                name=sn,
                lat=info["lat"],
                lon=info["lon"],
                risk_score=info["risk_score"],
                risk_label=info["risk_label"],
                risk_color=info["risk_color"],
                road_type=info["road_type"],
                mandal=info["mandal"],
                factors=info["factors"],
                is_hotspot=info["is_hotspot"],
                weather=W,
            )
            for sn, info in top5
        ]

        return RouteResponse(
            origin=source,
            destination=destination,
            weather=WeatherInfo(
                condition=W,
                temperature_c=str(w_info.get("temperature_c", "N/A")),
                rainfall_mm=float(w_info.get("rainfall_mm", 0)),
            ),
            model_name=self.loader.get_model_name(),
            model_accuracy=self.loader.get_model_accuracy(),
            routes=routes,
            recommendation=recommendation,
            top_risk_segments=top_risk_segs,
        )

    def get_heatmap(self, weather: str) -> List[HeatmapSegment]:
        IR = int(weather == "rainy")
        IF = int(weather == "foggy")
        IC = int(weather == "cloudy")
        _, seg_risks = self._build_network(weather, IR, IF, IC)
        return [
            HeatmapSegment(
                name=name,
                lat=info["lat"],
                lon=info["lon"],
                risk_score=info["risk_score"],
                risk_label=info["risk_label"],
                risk_color=info["risk_color"],
                road_type=info["road_type"],
                mandal=info["mandal"],
                is_hotspot=info["is_hotspot"],
            )
            for name, info in seg_risks.items()
        ]

    # ── Private ──────────────────────────────────────────────

    def _build_network(self, W: str, IR: int, IF: int, IC: int):
    loader = self.loader
    ws_map = {"foggy": 4, "rainy": 3, "cloudy": 2, "clear": 1}
    ws     = ws_map.get(W, 1)

    # Real mandal base risk — Peddapalli district calibrated
    MANDAL_BASE_RISK = {
        "Ramagundam": 0.72, "Manthani":  0.68,
        "Sultanabad": 0.65, "Palakurthy":0.62,
        "Ramagiri":   0.58, "Peddapalli":0.55,
        "Dharmaram":  0.50, "Anthergaon":0.45,
        "Srirampur":  0.42, "Kamanpur":  0.40,
        "Eligaid":    0.35, "Julapalli": 0.32,
        "Odela":      0.28, "Mutharam":  0.25,
    }

    seg_risk_base: Dict[str, float] = {}
    seg_risks: Dict[str, dict]      = {}

    for seg, info in loader.SEGMENT_DATA.items():
        h  = loader._hist_cnt_map.get(seg, 0)   # 0 = no recorded incidents
        m  = (loader._mandal_sev.get(info["mandal"])
              or MANDAL_BASE_RISK.get(info["mandal"], 0.35))
        rt = info["road_type"]
        rtr = ROAD_RISK_MAP.get(rt, 2)

        score = loader.predict_risk(
            road_type=rt, weather_cond=W, road_type_risk=rtr,
            hist_count=h, mandal_rl=m,
            is_rainy=IR, is_foggy=IF, is_cloudy=IC,
            weather_severity=ws,
            is_junction=int(rt == "junction"),
            is_highway=int(rt == "highway"),
            segment_name=seg,
        )
        seg_risk_base[seg] = score
        seg_risks[seg] = {
            "risk_score":  score,
            "risk_label":  _risk_label(score),
            "risk_color":  _risk_color(score),
            "road_type":   rt,
            "mandal":      info["mandal"],
            "lat":         info["lat"],
            "lon":         info["lon"],
            "factors":     _get_factors(rt, W, seg, h),
            "weather":     W,
            "hist_count":  h,
            "is_hotspot":  seg in REAL_HOTSPOTS,
        }

    graph: Dict[str, list] = defaultdict(list)
    for (u, v, km, tm, _rt) in loader.EDGE_LIST:
        avg_risk = (seg_risk_base.get(u, 0.4) + seg_risk_base.get(v, 0.4)) / 2
        graph[u].append((v, km, tm, avg_risk))
        graph[v].append((u, km, tm, avg_risk))

    return graph, seg_risks
    def _dijkstra(self, graph, source: str, target: str, weight: str = "risk"):
        # Must check against known keys explicitly — defaultdict auto-creates
        # keys on access, so 'x not in graph' would always be False
        known_nodes = set(graph.keys())
        if source not in known_nodes or target not in known_nodes:
            return [], []

        dist = defaultdict(lambda: float("inf"))
        dist[source] = 0
        # Store only the single incoming edge per node (O(n) memory, not O(n^2))
        prev: Dict[str, str]   = {}
        edge_data: Dict[str, tuple] = {}  # v -> (u, v, km, tm, rs)
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == target:
                break
            for (v, km, tm, rs) in graph[u]:
                if weight == "risk":
                    w = rs
                elif weight == "time":
                    w = tm / 200.0
                else:  # balanced
                    w = rs * 0.5 + (tm / 200.0) * 0.5
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    edge_data[v] = (u, v, km, tm, rs)
                    heapq.heappush(pq, (nd, v))

        # Reconstruct path
        path, node = [], target
        while node in prev:
            path.append(node)
            node = prev[node]
        path.append(source)
        path.reverse()

        if len(path) < 2 or path[-1] != target:
            return [], []

        # Reconstruct trail in O(n) by walking prev pointers
        trail = []
        node = target
        while node in prev:
            trail.append(edge_data[node])
            node = prev[node]
        trail.reverse()

        return path, trail