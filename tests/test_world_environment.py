import io
import json
from datetime import datetime, timezone

from world_environment import (
    CachedWeatherProvider,
    EnvironmentObservation,
    LocalSurfaceEnvironment,
    dew_point_c,
    parse_open_meteo_current,
)
from benchmarks.benchmark_world_environment import run as run_environment_benchmark


def observation(**changes):
    values = {
        "observed_at": datetime(2026, 8, 27, tzinfo=timezone.utc),
        "air_temperature_c": 10.0,
        "relative_humidity": 0.98,
        "precipitation_mm": 0.0,
        "cloud_cover": 0.0,
        "wind_speed_m_s": 0.2,
        "wind_direction_deg": 90.0,
        "shortwave_radiation_w_m2": 0.0,
        "source": "test",
        "fetched_at_monotonic": 0.0,
        "stale": False,
    }
    values.update(changes)
    return EnvironmentObservation(**values)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_open_meteo_parser_normalizes_units_and_provenance():
    parsed = parse_open_meteo_current(
        {
            "current": {
                "time": "2026-08-27T12:15",
                "temperature_2m": 18.5,
                "relative_humidity_2m": 81,
                "precipitation": 0.2,
                "cloud_cover": 63,
                "wind_speed_10m": 3.4,
                "wind_direction_10m": 370,
                "shortwave_radiation": 240,
            }
        },
        fetched_at_monotonic=12.0,
    )
    assert parsed.relative_humidity == 0.81
    assert parsed.wind_direction_deg == 10.0
    assert parsed.source == "open_meteo_current"
    assert parsed.stale is False


def test_weather_provider_caches_and_marks_last_evidence_stale_on_failure():
    calls = []
    clock = [100.0]
    payload = {
        "current": {
            "time": "2026-08-27T12:15Z",
            "temperature_2m": 17,
            "relative_humidity_2m": 70,
            "precipitation": 0,
            "cloud_cover": 30,
            "wind_speed_10m": 2,
            "wind_direction_10m": 180,
            "shortwave_radiation": 100,
        }
    }

    def opener(url, timeout):
        calls.append((url, timeout))
        if len(calls) > 1:
            raise OSError("offline")
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    provider = CachedWeatherProvider(opener=opener, clock=lambda: clock[0], cache_seconds=60)
    first = provider.current()
    assert provider.current() is first
    clock[0] = 161.0
    stale = provider.current()
    assert len(calls) == 2
    assert stale.stale is True
    assert stale.air_temperature_c == first.air_temperature_c


def test_dew_point_rises_with_humidity():
    assert dew_point_c(10.0, 0.95) > dew_point_c(10.0, 0.50)
    assert abs(dew_point_c(10.0, 1.0) - 10.0) < 0.01


def test_lazy_night_condensation_creates_unlabelled_surface_channels():
    environment = LocalSurfaceEnvironment()
    surface = environment.surfaces["garden_open"]
    surface.updated_at_monotonic = 0.0
    surface.surface_temperature_c = 6.0
    result = environment.observe("garden_open", observation(), now_monotonic=3600.0)
    assert result["channels"]["channel_0"] > 0.0
    assert set(result["channels"]) == {"channel_0", "channel_1", "channel_2"}
    assert "dew" not in json.dumps(result).lower()


def test_cloud_cover_reduces_night_radiative_cooling():
    clear = LocalSurfaceEnvironment()
    cloudy = LocalSurfaceEnvironment()
    clear_surface = clear.surfaces["garden_open"]
    cloudy_surface = cloudy.surfaces["garden_open"]
    for surface in (clear_surface, cloudy_surface):
        surface.updated_at_monotonic = 0.0
        surface.surface_temperature_c = 10.0
    clear_reading = clear.observe("garden_open", observation(cloud_cover=0.0), now_monotonic=3600.0)
    cloudy_reading = cloudy.observe("garden_open", observation(cloud_cover=1.0), now_monotonic=3600.0)
    assert clear_reading["channels"]["channel_1"] < cloudy_reading["channels"]["channel_1"]


def test_sun_and_airflow_dry_paving_faster_than_retaining_grass():
    environment = LocalSurfaceEnvironment()
    for surface in environment.surfaces.values():
        surface.updated_at_monotonic = 0.0
        surface.surface_temperature_c = 10.0
        surface.moisture_mm = 0.4
    drying = observation(
        air_temperature_c=20.0,
        relative_humidity=0.45,
        wind_speed_m_s=4.0,
        shortwave_radiation_w_m2=650.0,
    )
    grass = environment.observe("garden_shade", drying, now_monotonic=3600.0)
    paving = environment.observe("paving_open", drying, now_monotonic=3600.0)
    assert grass["channels"]["channel_0"] > paving["channels"]["channel_0"]


def test_cached_sensor_read_does_not_advance_surface_state():
    environment = LocalSurfaceEnvironment()
    surface = environment.surfaces["garden_open"]
    surface.updated_at_monotonic = 0.0
    surface.surface_temperature_c = 6.0
    surface.advance(observation(), now_monotonic=3600.0)
    before = (surface.moisture_mm, surface.updated_at_monotonic)
    assert surface.sensor_channels() == surface.sensor_channels()
    assert (surface.moisture_mm, surface.updated_at_monotonic) == before


def test_benchmark_reports_lazy_to_active_crossover_without_promoting_runtime():
    report = run_environment_benchmark(
        surface_counts=(3,), observation_rates=(1, 4), event_count=2, repeats=2
    )
    result = report["results"][0]
    assert set(result["cost_by_observation_rate"][0]) >= {
        "V2_lazy_on_observation", "V3_active_region"
    }
    assert result["measured_crossover"]["status"] in {"measured_candidate", "not_measured"}
    assert "requires repeated" in report["decision_boundary"]
