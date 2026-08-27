"""Low-cadence external weather forcing and lazy surface-moisture mechanics.

Weather is evidence about the physical setting, not a semantic description for
Ina.  The surface sensor therefore exposes physical channels and provenance but
does not label condensation as "dew".
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Dict, Mapping, Optional


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class EnvironmentObservation:
    observed_at: datetime
    air_temperature_c: float
    relative_humidity: float
    precipitation_mm: float
    cloud_cover: float
    wind_speed_m_s: float
    wind_direction_deg: float
    shortwave_radiation_w_m2: float
    source: str
    fetched_at_monotonic: float
    stale: bool = False

    def public_channels(self) -> Dict[str, object]:
        """Inspectability record for world systems, separate from sensory data."""
        return {
            "observed_at": self.observed_at.isoformat(),
            "air_temperature_c": round(self.air_temperature_c, 3),
            "relative_humidity": round(self.relative_humidity, 4),
            "precipitation_mm": round(self.precipitation_mm, 4),
            "cloud_cover": round(self.cloud_cover, 4),
            "wind_speed_m_s": round(self.wind_speed_m_s, 4),
            "wind_direction_deg": round(self.wind_direction_deg, 2),
            "shortwave_radiation_w_m2": round(self.shortwave_radiation_w_m2, 3),
            "source": self.source,
            "stale": self.stale,
        }


def default_environment_observation(*, now_monotonic: Optional[float] = None) -> EnvironmentObservation:
    """Conservative offline forcing; clearly marked as a synthetic fallback."""
    return EnvironmentObservation(
        observed_at=datetime.now(timezone.utc),
        air_temperature_c=12.0,
        relative_humidity=0.70,
        precipitation_mm=0.0,
        cloud_cover=0.50,
        wind_speed_m_s=1.0,
        wind_direction_deg=0.0,
        shortwave_radiation_w_m2=0.0,
        source="offline_fallback",
        fetched_at_monotonic=time.monotonic() if now_monotonic is None else now_monotonic,
        stale=True,
    )


def parse_open_meteo_current(payload: Mapping[str, object], *, fetched_at_monotonic: float) -> EnvironmentObservation:
    current = payload.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("weather response has no current observation")

    def number(key: str, default: Optional[float] = None) -> float:
        value = current.get(key, default)
        if value is None or isinstance(value, bool):
            raise ValueError(f"weather response has no numeric {key}")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"weather response has invalid {key}")
        return parsed

    observed_text = str(current.get("time") or "")
    try:
        observed_at = datetime.fromisoformat(observed_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("weather response has invalid observation time") from exc
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)

    return EnvironmentObservation(
        observed_at=observed_at,
        air_temperature_c=number("temperature_2m"),
        relative_humidity=_clamp(number("relative_humidity_2m") / 100.0, 0.0, 1.0),
        precipitation_mm=max(0.0, number("precipitation", 0.0)),
        cloud_cover=_clamp(number("cloud_cover", 0.0) / 100.0, 0.0, 1.0),
        wind_speed_m_s=max(0.0, number("wind_speed_10m", 0.0)),
        wind_direction_deg=number("wind_direction_10m", 0.0) % 360.0,
        shortwave_radiation_w_m2=max(0.0, number("shortwave_radiation", 0.0)),
        source="open_meteo_current",
        fetched_at_monotonic=fetched_at_monotonic,
        stale=False,
    )


class CachedWeatherProvider:
    """Fetch once when explicitly requested; callers decide when evidence is stale."""

    def __init__(
        self,
        *,
        latitude: float = 51.48,
        longitude: float = -0.03,
        cache_seconds: float = 3600.0,
        timeout_seconds: float = 4.0,
        opener: Callable[..., object] = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.cache_seconds = max(60.0, float(cache_seconds))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._opener = opener
        self._clock = clock
        self._cached: Optional[EnvironmentObservation] = None

    def current(self, *, force: bool = False) -> EnvironmentObservation:
        now = self._clock()
        if self._cached is not None and not force:
            age = now - self._cached.fetched_at_monotonic
            if age < self.cache_seconds:
                return self._cached
        query = urllib.parse.urlencode({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,"
                "wind_speed_10m,wind_direction_10m,shortwave_radiation"
            ),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        })
        try:
            with self._opener(f"{OPEN_METEO_URL}?{query}", timeout=self.timeout_seconds) as response:
                payload = json.load(response)
            observation = parse_open_meteo_current(payload, fetched_at_monotonic=now)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError):
            if self._cached is not None:
                observation = replace(self._cached, stale=True)
            else:
                observation = default_environment_observation(now_monotonic=now)
        self._cached = observation
        return observation


def dew_point_c(air_temperature_c: float, relative_humidity: float) -> float:
    """Magnus approximation for water over ordinary environmental temperatures."""
    humidity = _clamp(relative_humidity, 1e-6, 1.0)
    alpha = math.log(humidity) + (17.625 * air_temperature_c) / (243.04 + air_temperature_c)
    return 243.04 * alpha / (17.625 - alpha)


@dataclass(frozen=True)
class SurfaceMaterial:
    name: str
    solar_absorption: float
    moisture_capacity_mm: float
    retention: float
    night_cooling_c: float


@dataclass
class SurfaceState:
    material: SurfaceMaterial
    solar_exposure: float = 1.0
    moisture_mm: float = 0.0
    surface_temperature_c: Optional[float] = None
    updated_at_monotonic: Optional[float] = None

    def sensor_channels(self) -> Dict[str, float]:
        """Read the last resolved state without advancing environmental time."""
        temperature = self.surface_temperature_c
        saturation = self.moisture_mm / max(self.material.moisture_capacity_mm, 1e-9)
        return {
            "channel_0": round(saturation, 6),
            "channel_1": round(temperature, 4) if temperature is not None else 0.0,
            "channel_2": round(min(1.0, saturation * (1.0 - self.material.retention * 0.45)), 6),
        }

    def advance(self, observation: EnvironmentObservation, *, now_monotonic: float) -> Dict[str, float]:
        """Resolve elapsed change only when observed or environmental evidence changes."""
        if self.updated_at_monotonic is None:
            elapsed_hours = 0.0
        else:
            elapsed_hours = _clamp((now_monotonic - self.updated_at_monotonic) / 3600.0, 0.0, 6.0)
        self.updated_at_monotonic = now_monotonic

        solar = observation.shortwave_radiation_w_m2 * _clamp(self.solar_exposure, 0.0, 1.0)
        # Clear surfaces radiate more heat to the night sky; cloud cover insulates.
        sky_exposure = 1.0 - (0.75 * observation.cloud_cover)
        radiative_cooling = self.material.night_cooling_c * sky_exposure if solar < 5.0 else 0.0
        solar_warming = min(12.0, solar * self.material.solar_absorption / 85.0)
        target_temperature = observation.air_temperature_c - radiative_cooling + solar_warming
        if self.surface_temperature_c is None:
            self.surface_temperature_c = target_temperature
        else:
            blend = 1.0 - math.exp(-max(elapsed_hours, 1.0 / 60.0) * 1.5)
            self.surface_temperature_c += (target_temperature - self.surface_temperature_c) * blend

        point = dew_point_c(observation.air_temperature_c, observation.relative_humidity)
        condensation_margin = max(0.0, point - self.surface_temperature_c)
        condensation = condensation_margin * 0.035 * elapsed_hours
        precipitation = observation.precipitation_mm * min(1.0, max(elapsed_hours, 1.0 / 60.0))
        evaporation = (
            (solar / 700.0)
            + observation.wind_speed_m_s * 0.018
            + max(0.0, self.surface_temperature_c - point) * 0.012
        ) * (1.1 - 0.8 * self.material.retention) * elapsed_hours
        self.moisture_mm = _clamp(
            self.moisture_mm + condensation + precipitation - evaporation,
            0.0,
            self.material.moisture_capacity_mm,
        )
        return self.sensor_channels()


class LocalSurfaceEnvironment:
    """Small material substrate for V1; no whole-world grid or continuous tick."""

    def __init__(self) -> None:
        grass = SurfaceMaterial("grass", 0.72, 1.8, 0.90, 3.2)
        paving = SurfaceMaterial("paving", 0.82, 0.45, 0.18, 2.3)
        self.surfaces: Dict[str, SurfaceState] = {
            "garden_open": SurfaceState(grass, solar_exposure=1.0),
            "garden_shade": SurfaceState(grass, solar_exposure=0.20),
            "paving_open": SurfaceState(paving, solar_exposure=1.0),
        }

    def observe(self, surface_id: str, observation: EnvironmentObservation, *, now_monotonic: float) -> Dict[str, object]:
        surface = self.surfaces[surface_id]
        channels = surface.advance(observation, now_monotonic=now_monotonic)
        return {
            "surface_id": surface_id,
            "channels": channels,
            "observed_at": observation.observed_at.isoformat(),
            "forcing_source": observation.source,
            "forcing_stale": observation.stale,
        }
