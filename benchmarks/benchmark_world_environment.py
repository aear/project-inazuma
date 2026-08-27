"""Versioned cost and crossover benchmark for world surface moisture modes."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

from world_environment import EnvironmentObservation, LocalSurfaceEnvironment


def _observation(*, solar: float, humidity: float) -> EnvironmentObservation:
    return EnvironmentObservation(
        observed_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
        air_temperature_c=11.0,
        relative_humidity=humidity,
        precipitation_mm=0.0,
        cloud_cover=0.2,
        wind_speed_m_s=0.8,
        wind_direction_deg=225.0,
        shortwave_radiation_w_m2=solar,
        source="deterministic_benchmark",
        fetched_at_monotonic=0.0,
    )


def _measure(function, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples), 6),
        "min_ms": round(min(samples), 6),
    }


def _make_surfaces(count: int):
    environments = [LocalSurfaceEnvironment() for _ in range((count + 2) // 3)]
    surfaces = []
    for environment in environments:
        surfaces.extend(environment.surfaces.values())
    return surfaces[:count]


def _prepare_surfaces(count: int):
    surfaces = _make_surfaces(count)
    for surface in surfaces:
        surface.updated_at_monotonic = 0.0
        surface.surface_temperature_c = 6.0
    return surfaces


def _lazy_workload(count: int, observations_per_surface: int, event_count: int, weather) -> float:
    surfaces = _prepare_surfaces(count)
    checksum = 0.0
    for event in range(1, event_count + 1):
        now = event * 900.0
        for surface in surfaces:
            for _ in range(observations_per_surface):
                checksum += surface.advance(weather, now_monotonic=now)["channel_0"]
    return checksum


def _active_workload(count: int, observations_per_surface: int, event_count: int, weather) -> float:
    surfaces = _prepare_surfaces(count)
    checksum = 0.0
    for event in range(1, event_count + 1):
        now = event * 900.0
        for surface in surfaces:
            surface.advance(weather, now_monotonic=now)
        for surface in surfaces:
            for _ in range(observations_per_surface):
                checksum += surface.sensor_channels()["channel_0"]
    return checksum


def _crossover(rows: list[dict]) -> dict:
    active_wins = [
        row for row in rows
        if row["V3_active_region"]["median_ms"] < row["V2_lazy_on_observation"]["median_ms"]
    ]
    if not active_wins:
        return {"observations_per_surface_per_event": None, "status": "not_measured"}
    winner = min(active_wins, key=lambda row: row["observations_per_surface_per_event"])
    return {
        "observations_per_surface_per_event": winner["observations_per_surface_per_event"],
        "status": "measured_candidate",
        "rule": "activate only after repeated bounded measurements near this workload",
    }


def run(
    *,
    surface_counts=(3, 30, 300),
    observation_rates=(1, 2, 4, 8, 16),
    event_count: int = 4,
    repeats: int = 20,
) -> dict:
    night = _observation(solar=0.0, humidity=0.98)
    results = []
    for count in surface_counts:
        metadata = [night.public_channels() for _ in range(count)]
        baseline = _measure(lambda: sum(float(row["relative_humidity"]) for row in metadata), repeats)

        causal_signal = _lazy_workload(count, 1, 1, night)
        cost_rows = []
        for observation_rate in observation_rates:
            lazy = _measure(
                lambda rate=observation_rate: _lazy_workload(count, rate, event_count, night),
                repeats,
            )
            active = _measure(
                lambda rate=observation_rate: _active_workload(count, rate, event_count, night),
                repeats,
            )
            cost_rows.append({
                "observations_per_surface_per_event": observation_rate,
                "environment_events": event_count,
                "V2_lazy_on_observation": lazy,
                "V3_active_region": active,
            })
        results.append({
            "surface_count": count,
            "V1_metadata_only": baseline,
            "V1_can_expose_condensation_difference": False,
            "V2_condensation_channel_sum": round(causal_signal, 6),
            "cost_by_observation_rate": cost_rows,
            "measured_crossover": _crossover(cost_rows),
        })
    return {
        "benchmark": "world environment causal substrate",
        "versions": {
            "V1": "weather metadata only; retained zero-mechanics baseline",
            "V2": "lazy moisture and temperature resolution on each observation",
            "V3": "candidate active region; resolve once per environment event, then read cached channels",
        },
        "input": "deterministic synthetic weather; no network or Ina memory",
        "decision_boundary": (
            "Benchmark evidence only. Runtime promotion requires repeated operation-local "
            "measurements and should use hysteresis rather than one crossover sample."
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-counts", default="3,30,300")
    parser.add_argument("--observation-rates", default="1,2,4,8,16")
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    counts = tuple(max(1, int(value)) for value in args.surface_counts.split(",") if value.strip())
    rates = tuple(max(1, int(value)) for value in args.observation_rates.split(",") if value.strip())
    print(json.dumps(run(
        surface_counts=counts,
        observation_rates=rates,
        event_count=max(1, args.events),
        repeats=max(1, args.repeats),
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
