import io
import json

from lyric_search_engine import search_lyrics, search_local_music_manifest, search_wikisource
from cognition_runtime.default_capabilities import build_task_profiles


class _Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *_args): self.close()


def test_lyric_search_benchmark_v1_ungated_scrape_vs_v2_candidate_only(tmp_path):
    payload = ["query", ["John Henry"], ["Traditional song"], ["https://en.wikisource.org/wiki/John_Henry"]]
    results = search_wikisource(
        "John Henry", opener=lambda _request, timeout: _Response(json.dumps(payload).encode())
    )
    assert results[0]["rights_status"] == "candidate_requires_page_license_review"
    assert results[0]["lyrics_ingestion_allowed"] is False


def test_local_search_exposes_verified_audio_but_not_unverified_lyrics(tmp_path):
    manifest = {"tracks": {"musicbox/excerpt.mp3": {
        "title": "John Henry", "creator": "Example Singer", "source_url": "https://loc.gov/item",
        "license_id": "LOC-MUSICBOX-FREE-TO-USE",
    }}}
    (tmp_path / "ina_public_music_manifest.json").write_text(json.dumps(manifest))
    result = search_local_music_manifest(tmp_path, "John Henry")
    assert result[0]["rights_status"] == "audio_verified_lyrics_not_present"
    assert result[0]["lyrics_ingestion_allowed"] is False


def test_search_remains_useful_when_web_provider_fails(tmp_path):
    result = search_lyrics(
        "John Henry", public_music_root=tmp_path,
        web_search=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    assert result["status"] == "partial"
    assert result["errors"] == [{"provider": "wikisource", "error": "TimeoutError"}]


def test_lyric_search_has_a_bounded_scheduler_profile():
    profile = build_task_profiles("Ina")["lyric_search_run"]
    assert profile["command"] == ["python", "lyric_search_engine.py", "--runtime-request", "--child", "Ina"]
    assert profile["memory_class"] == "low"
    assert profile["exclusive_group"] == "network_lookup"
