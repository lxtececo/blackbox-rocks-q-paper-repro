"""Geographic primitives for the fleet vertical.

Distances are great-circle (haversine) kilometres. Road networks are
approximated with a configurable detour factor so the classical baseline
and the quantum encodings share one consistent cost model — the whole
point of the comparison is that only the *assignment* differs.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0088

#: Straight-line → road distance multiplier (typical US regional value).
DEFAULT_DETOUR_FACTOR = 1.25

MILES_PER_KM = 0.621371


def km_to_miles(km: float) -> float:
    return float(km) * MILES_PER_KM


def miles_to_km(miles: float) -> float:
    return float(miles) / MILES_PER_KM


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def road_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    detour_factor: float = DEFAULT_DETOUR_FACTOR,
) -> float:
    """Haversine distance inflated by a road detour factor."""
    return haversine_km(lat1, lon1, lat2, lon2) * float(detour_factor)


def travel_minutes(distance_km: float, speed_kmh: float, traffic_factor: float = 1.0) -> float:
    """Drive time for ``distance_km`` at ``speed_kmh`` under congestion."""
    speed = max(float(speed_kmh), 1e-6)
    return (float(distance_km) / speed) * 60.0 * max(float(traffic_factor), 1e-6)


class DistanceOracle:
    """Lazily-memoised point-to-point distance lookup.

    620 delivery points + 3 depots is a 623x623 matrix (~388k entries).
    Materialising it eagerly costs ~3 MB and a noticeable startup burn on
    every replan, so entries are cached on first use instead — routes only
    ever touch a sparse slice of the matrix.
    """

    def __init__(
        self,
        points: Iterable[Tuple[str, float, float]],
        *,
        detour_factor: float = DEFAULT_DETOUR_FACTOR,
    ) -> None:
        self._coords: Dict[str, Tuple[float, float]] = {
            pid: (float(lat), float(lon)) for pid, lat, lon in points
        }
        self._detour = float(detour_factor)
        self._cache: Dict[Tuple[str, str], float] = {}

    def has(self, point_id: str) -> bool:
        return point_id in self._coords

    def coords(self, point_id: str) -> Tuple[float, float]:
        return self._coords[point_id]

    def km(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            lat1, lon1 = self._coords[a]
            lat2, lon2 = self._coords[b]
        except KeyError as exc:  # pragma: no cover — programming error
            raise KeyError(f"unknown point in distance oracle: {exc}") from exc
        value = road_km(lat1, lon1, lat2, lon2, self._detour)
        self._cache[key] = value
        return value

    def path_km(self, sequence: Sequence[str]) -> float:
        """Total distance along an ordered sequence of point ids."""
        return sum(
            self.km(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)
        )


def centroid(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Mean lat/lon of a small cluster (adequate at regional scale)."""
    if not points:
        return (0.0, 0.0)
    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    return (lat, lon)


def bounding_box(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """(min_lat, min_lon, max_lat, max_lon) — used for the SVG map viewport."""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    lats: List[float] = [p[0] for p in points]
    lons: List[float] = [p[1] for p in points]
    return (min(lats), min(lons), max(lats), max(lons))
