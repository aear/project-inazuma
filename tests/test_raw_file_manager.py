import json
import wave
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


from transformers.fractal_multidimensional_transformers import FractalTransformer


import raw_file_manager as rfm


def _create_wav_file(path: Path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00" * 200)


def _create_epub(path: Path, chapters, *, spine_order=None, extra_members=None):
    chapter_names = [name for name, _text in chapters]
    order = spine_order or chapter_names
    manifest_ids = {name: f"chapter_{index}" for index, name in enumerate(chapter_names)}
    manifest = "\n".join(
        f'<item id="{manifest_ids[name]}" href="{name}" media-type="application/xhtml+xml"/>'
        for name in chapter_names
    )
    spine = "\n".join(
        f'<itemref idref="{manifest_ids[name]}"/>' for name in order
    )
    container = (
        '<?xml version="1.0"?>'
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="EPUB/content.opf"/></rootfiles>'
        '</container>'
    )
    package = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f'<manifest>{manifest}</manifest><spine>{spine}</spine>'
        '</package>'
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("EPUB/content.opf", package)
        for name, body in chapters:
            archive.writestr(
                f"EPUB/{name}",
                (
                    '<html xmlns="http://www.w3.org/1999/xhtml">'
                    '<head><title>Hidden head title</title></head>'
                    f'<body><p>{body}</p></body></html>'
                ),
            )
        for name, data in extra_members or []:
            archive.writestr(name, data)


def test_epub_is_read_as_spine_ordered_document(tmp_path):
    epub_path = tmp_path / "novel.epub"
    _create_epub(
        epub_path,
        [
            ("first.xhtml", "First chapter &amp; opening."),
            ("second.xhtml", "Second chapter comes later."),
        ],
        spine_order=["second.xhtml", "first.xhtml"],
        extra_members=[
            (
                "EPUB/unlisted.xhtml",
                "<html><body>Unlisted appendix should not be read.</body></html>",
            )
        ],
    )

    text = rfm._extract_epub_text(epub_path)

    assert rfm.classify_path(epub_path) == "document"
    assert text.index("Second chapter") < text.index("First chapter")
    assert "First chapter & opening." in text
    assert "Hidden head title" not in text
    assert "Unlisted appendix" not in text
    assert rfm._extract_epub_text_bytes(epub_path.read_bytes(), "novel.epub") == text
    assert "\n\n" in text

    class Transformer:
        def encode(self, fragment):
            return {"importance": 0.5}

    fragments = rfm.fragment_document(epub_path, Transformer())
    assert fragments
    assert all("epub" in fragment["tags"] for fragment in fragments)
    assert all(fragment["modality"] == "text" for fragment in fragments)
    assert all(fragment["written_example"]["complete_text"] for fragment in fragments)
    assert all(fragment["written_example"]["interpretation_unit"] for fragment in fragments)
    assert all(fragment["written_example"]["storage_fragment"] for fragment in fragments)
    assert all(not fragment["written_example"]["transport_chunk"] for fragment in fragments)


    assert all("partial_document_read" in fragment["tags"] for fragment in fragments)
    assert all(
        fragment["document_read_progress"]["complete"] is False
        for fragment in fragments
    )
    assert all(
        fragment["document_read_progress"]["window_reaches_end"] is True
        for fragment in fragments
    )


def test_written_passages_keep_whole_words_and_paragraphs():
    text = (
        "First paragraph has a complete sentence. " * 12
        + "\n\n"
        + "Second paragraph also stays readable. " * 12
    )
    passages = rfm._written_passages(text, target_chars=240)

    assert len(passages) > 2
    assert all(len(passage) <= 240 for passage in passages)
    assert all(not passage.startswith(("irst", "econd")) for passage in passages)
    assert all(passage[-1] in ".!?" for passage in passages)
    assert " ".join(" ".join(passages).split()) == " ".join(text.split())


def test_document_passage_selection_and_ids_are_stable_for_rereads(monkeypatch):
    text = "\n\n".join(
        f"Paragraph {index} contains a complete example sentence. " * 8
        for index in range(30)
    )
    first = rfm._document_chunks(text, "same-book.epub", chunk_size=240, max_chunks=5)
    second = rfm._document_chunks(text, "same-book.epub", chunk_size=240, max_chunks=5)
    assert first == second

    class Transformer:
        def encode(self, fragment):
            return {"importance": 0.5}

    monkeypatch.setattr(rfm, "update_text_vocab", lambda *args, **kwargs: True)
    first_fragments = rfm.fragment_document_text(
        "\n\n".join(first), "same-book.epub", Transformer(), sequential=True
    )
    second_fragments = rfm.fragment_document_text(
        "\n\n".join(first), "same-book.epub", Transformer(), sequential=True
    )
    assert [item["written_example"]["passage_id"] for item in first_fragments] == [
        item["written_example"]["passage_id"] for item in second_fragments
    ]


def test_epub_tolerates_imperfect_xhtml(tmp_path):
    epub_path = tmp_path / "imperfect.epub"
    _create_epub(
        epub_path,
        [("chapter.xhtml", "Readable <b>words</p> remain")],
    )

    text = rfm._extract_epub_text(epub_path)

    assert "Readable words remain" in text


def test_epub_rejects_unsafe_or_unbounded_packages(monkeypatch, tmp_path):
    assert rfm._safe_epub_member_name("EPUB/content.opf", "../../outside.xhtml") is None

    epub_path = tmp_path / "bounded.epub"
    _create_epub(epub_path, [("chapter.xhtml", "A safe chapter.")])
    monkeypatch.setattr(rfm, "EPUB_ENTRY_COUNT_LIMIT", 2)

    assert rfm._extract_epub_text(epub_path) == ""


def test_epub_cursor_progresses_until_the_whole_book_is_read(tmp_path):
    epub_path = tmp_path / "long.epub"
    _create_epub(
        epub_path,
        [
            ("first.xhtml", "A" * 700),
            ("second.xhtml", "B" * 700),
        ],
    )

    cursor = None
    windows = []
    for _pass in range(4):
        text, progress = rfm._extract_epub_text(
            epub_path,
            max_chars=500,
            cursor=cursor,
            with_progress=True,
        )
        windows.append(text)
        if progress["complete"]:
            break
        cursor = progress["next_cursor"]

    assert len(windows) == 3
    assert all(len(window) <= 500 for window in windows)
    assert progress["complete"] is True
    assert cursor == {"section": 1, "char": 298}
    assert "".join(windows).replace("\n", "") == ("A" * 700) + ("B" * 700)


def test_epub_document_cursor_keeps_history_resumable():
    stamp = {"mtime_ns": 10, "size_bytes": 20}
    record = dict(stamp)

    rfm._set_self_read_continuation(
        record,
        stamp,
        next_offset=30,
        total_fragments=30,
        document_cursor={"section": 2, "char": 125},
    )

    assert rfm.classify_self_read_file(record, stamp) == "resume"
    assert rfm._self_read_resume_offset(record, stamp) == 0
    assert rfm._epub_cursor_from_history(record, stamp) == {
        "section": 2,
        "char": 125,
    }

    rfm._set_self_read_continuation(
        record,
        stamp,
        next_offset=30,
        total_fragments=30,
    )
    assert "continuation" not in record


def test_fragment_audio_wav(tmp_path):
    wav_path = tmp_path / "sample.wav"
    _create_wav_file(wav_path)

    transformer = FractalTransformer()
    fragments = rfm.fragment_audio(wav_path, transformer)

    assert fragments, "Expected a fragment for WAV input"
    frag = fragments[0]
    assert frag["modality"] == "audio"
    assert frag["source"] == str(wav_path)
    assert "importance" in frag
    assert "self_read" in frag["tags"]


def test_fragment_audio_opus_flattens_frames_and_keeps_metadata(
    monkeypatch,
    tmp_path,
):
    opus_path = tmp_path / "clip.opus"
    opus_path.write_bytes(b"fake opus data")

    analysis = {
        "summary": "Synthetic compressed-audio analysis",
        "clarity": 0.432187,
        "tags": ["audio", "digest", "synthetic"],
        "emotions": {"focus": 0.2},
        "frames": [[-1.0, -0.5], [-0.25, 0.0]],
        "texture_signature": {"rms": 0.2},
        "language_hint": "en",
    }
    metadata = {
        "title": "Way Back Home",
        "artist": ["projectgodhunter", "Ina"],
        "lyrics": "A short embedded lyric.",
        "technical": {
            "codec": "opus",
            "sample_rate": 48000,
            "attached_picture": True,
        },
    }

    monkeypatch.setattr(rfm, "analyze_audio_clip", lambda path, transformer: analysis)
    monkeypatch.setattr(rfm, "_extract_audio_metadata", lambda path: metadata)

    transformer = FractalTransformer()
    fragments = rfm.fragment_audio(opus_path, transformer)

    assert fragments, "Expected a fragment for compressed audio input"
    frag = fragments[0]
    assert frag["modality"] == "audio"
    assert frag["audio_features"] == [-1.0, -0.5, -0.25, 0.0]
    assert all(isinstance(value, float) for value in frag["audio_features"])
    assert frag["audio_metadata"] == metadata
    assert frag["summary"].startswith("Way Back Home by projectgodhunter; Ina.")
    assert frag["audio_analysis"]["frame_count"] == 2
    assert frag["audio_analysis"]["feature_bins"] == 2
    assert "self_read" in frag["tags"]
    assert "synthetic" in frag["tags"]
    assert "audio_metadata" in frag["tags"]
    assert "embedded_lyrics" in frag["tags"]
    assert "embedded_cover_art" in frag["tags"]
    assert frag["importance"] == pytest.approx(0.4322, rel=0, abs=1e-4)


def test_audio_metadata_normalizes_multivalue_tags_and_audio_stream(monkeypatch):
    monkeypatch.setattr(
        rfm,
        "mediainfo_json",
        lambda _path: {
            "format": {
                "format_name": "mp3",
                "duration": ["287.424", "fallback"],
                "size": "6666807",
                "tags": {
                    "TITLE": ["Way Back Home", "Alternate title"],
                    "artist": "projectgodhunter",
                    "comment": "created by Ina",
                    "lyrics-eng": ["First line", "Second line"],
                    "custom-tag": ["one", "two"],
                },
            },
            "streams": [
                {
                    "index": 1,
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "disposition": {"attached_pic": 1},
                },
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "sample_rate": ["48000"],
                    "channels": 2,
                    "bit_rate": "185130",
                    "disposition": {"attached_pic": 0},
                },
            ],
        },
    )

    metadata = rfm._extract_audio_metadata(Path("song.mp3"))

    assert metadata["title"] == ["Way Back Home", "Alternate title"]
    assert metadata["artist"] == "projectgodhunter"
    assert metadata["lyrics"] == "First line\n\nSecond line"
    assert metadata["tags"]["custom_tag"] == ["one", "two"]
    assert metadata["technical"] == {
        "format": "mp3",
        "codec": "mp3",
        "duration_seconds": 287.424,
        "bit_rate": 185130,
        "sample_rate": 48000,
        "channels": 2,
        "file_size": 6666807,
        "stream_index": 0,
        "attached_picture": True,
        "artwork_codec": "mjpeg",
    }


def test_rapidcrest_music_is_external_signed_artist(tmp_path):
    fragment = {"tags": ["self_read"], "metadata": {"flags": []}}

    rfm.annotate_fragment_source(
        fragment,
        "music",
        "Rapidcrest: Da Drums Beat/master.wav",
        tmp_path,
    )

    assert "external_music" in fragment["tags"]
    assert "signed_artist" in fragment["tags"]
    assert "ina_music" not in fragment["tags"]
    assert "self_voice" not in fragment["tags"]
    assert fragment["provenance"] == "signed_artist_catalog"
    assert fragment["source_context"]["ownership_hint"] == "external_artist"
    assert fragment["source_context"]["external_artist_hint"] == "rapidcrest"


@pytest.mark.parametrize(
    "relative_path",
    [
        "Ina Sings: Soft Orbit/master.wav",
        "Ina Sings_ Way Back Home/Way Back Home.mp3",
    ],
)
def test_ina_music_keeps_self_voice_ownership(tmp_path, relative_path):
    fragment = {"tags": ["self_read"], "metadata": {"flags": []}}

    rfm.annotate_fragment_source(
        fragment,
        "music",
        relative_path,
        tmp_path,
    )

    assert "ina_music" in fragment["tags"]
    assert "self_voice" in fragment["tags"]
    assert "external_music" not in fragment["tags"]
    assert fragment["source_context"]["ownership_hint"] == "self_voice"


def test_read_history_migrates_legacy_list_to_versioned_ledger(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    history_path = (
        tmp_path / "AI_Children" / "Ina" / "memory" / "read_history.json"
    )
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(["Project Inazuma/code.py", "legacy.py"]),
        encoding="utf-8",
    )

    ledger = rfm.load_history("Ina")

    assert ledger["version"] == rfm.SELF_READ_HISTORY_VERSION
    assert ledger["migration"]["from"] == "legacy_string_list"
    assert ledger["files"]["Project Inazuma/code.py"]["read_count"] == 1
    assert ledger["files"]["legacy.py"]["legacy_migrated"] is True

    ledger["last_pass"] = {"read_focus": "balanced"}
    rfm.save_history("Ina", ledger)
    stored = json.loads(history_path.read_text(encoding="utf-8"))

    assert stored["version"] == rfm.SELF_READ_HISTORY_VERSION
    assert list(stored["files"]) == sorted(stored["files"])
    assert stored["last_pass"] == {"read_focus": "balanced"}
    assert rfm.load_history("Ina")["last_pass"] == {"read_focus": "balanced"}


def test_history_key_uses_source_and_resolved_root_and_migrates_old_key(tmp_path):
    root_a = tmp_path / "one" / "shared"
    root_b = tmp_path / "two" / "shared"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)

    key_a = rfm.self_read_history_key("code", root_a, "same.py")
    key_b = rfm.self_read_history_key("code", root_b, "same.py")
    key_music = rfm.self_read_history_key("music", root_a, "same.py")

    assert key_a != key_b
    assert key_a != key_music

    old_key = "shared/same.py"
    files = {old_key: rfm._legacy_history_record()}
    canonical, prior = rfm._resolve_history_record(
        files,
        source_key="code",
        base_root=root_a,
        relative_path="same.py",
    )

    assert canonical == key_a
    assert old_key not in files
    assert files[key_a]["migrated_from_key"] == old_key
    assert prior is files[key_a]


def test_history_resolver_recognizes_legacy_basename(tmp_path):
    files = {"same.py": rfm._legacy_history_record()}

    canonical, prior = rfm._resolve_history_record(
        files,
        source_key="code",
        base_root=tmp_path,
        relative_path="nested/same.py",
        allow_legacy_basename=True,
    )

    assert "same.py" not in files
    assert prior["migrated_from_key"] == "same.py"
    assert files[canonical] is prior


def test_stat_fingerprint_detects_updated_seen_file():
    prior = {"mtime_ns": 100, "size_bytes": 10, "read_count": 1}

    assert rfm.classify_self_read_file(None, {"mtime_ns": 100, "size_bytes": 10}) == "new"
    assert rfm.classify_self_read_file(prior, {"mtime_ns": 100, "size_bytes": 10}) is None
    assert (
        rfm.classify_self_read_file(prior, {"mtime_ns": 101, "size_bytes": 10})
        == "updated"
    )
    assert (
        rfm.classify_self_read_file(prior, {"mtime_ns": 100, "size_bytes": 11})
        == "updated"
    )


def test_revisit_selection_is_oldest_first_bounded_and_focus_aware():
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)

    def candidate(name, age_hours):
        return {
            "history_key": name,
            "prior": {
                "last_read_at": (now - timedelta(hours=age_hours)).isoformat(),
            },
        }

    candidates = [
        candidate("newest", 1),
        candidate("oldest", 20),
        candidate("middle", 10),
        candidate("older", 15),
    ]

    selected = rfm.select_revisit_candidates(
        candidates,
        "seen",
        limit=2,
        now_ts=now.timestamp(),
        min_age_seconds=2 * 3600,
    )
    balanced = rfm.select_revisit_candidates(
        candidates,
        "balanced",
        limit=5,
        now_ts=now.timestamp(),
        min_age_seconds=0,
    )

    assert [item["history_key"] for item in selected] == ["oldest", "older"]
    assert [item["history_key"] for item in balanced] == ["oldest"]
    assert rfm.select_revisit_candidates(candidates, "new", limit=5) == []


def test_fragment_read_annotation_preserves_prior_lineage():
    fragment = {"tags": ["self_read"], "source_context": {}}
    prior = {
        "last_read_at": "2026-07-30T10:00:00+00:00",
        "read_count": 2,
        "last_read_reason": "updated",
        "mtime_ns": 100,
        "size_bytes": 20,
        "last_fragment_ids": ["frag_old"],
    }
    record = {
        "read_count": 3,
        "mtime_ns": 100,
        "size_bytes": 20,
    }

    rfm.annotate_fragment_read_lineage(
        fragment,
        read_reason="revisit",
        prior=prior,
        record=record,
        focus="seen",
    )

    context = fragment["source_context"]
    assert "self_read_revisit" in fragment["tags"]
    assert context["read_reason"] == "revisit"
    assert context["read_focus"] == "seen"
    assert context["read_count"] == 3
    assert context["prior_read"]["read_count"] == 2
    assert context["prior_read"]["fragment_ids"] == ["frag_old"]


def test_manual_focus_resolution_uses_emotion_then_balanced_default(monkeypatch):
    monkeypatch.delenv(rfm.SELF_READ_FOCUS_ENV, raising=False)
    monkeypatch.setattr(
        rfm,
        "_load_self_read_emotion_values",
        lambda child: {
            "values": {
                "familiarity": 0.9,
                "fuzziness": 0.8,
                "clarity": 0.1,
            }
        },
    )

    emotional = rfm.resolve_self_read_focus("Ina")

    assert emotional["focus"] == "seen"
    assert emotional["source"] == "emotion_state"
    assert emotional["seen_score"] > emotional["new_score"]

    monkeypatch.setattr(rfm, "_load_self_read_emotion_values", lambda child: {})
    assert rfm.resolve_self_read_focus("Ina")["focus"] == "balanced"


def test_environment_focus_override_keeps_emotion_scores(monkeypatch):
    monkeypatch.setenv(rfm.SELF_READ_FOCUS_ENV, "new")
    monkeypatch.setattr(
        rfm,
        "_load_self_read_emotion_values",
        lambda child: {"values": {"familiarity": 0.9, "clarity": 0.1}},
    )

    decision = rfm.resolve_self_read_focus("Ina")

    assert decision["focus"] == "new"
    assert decision["suggested_focus"] == "seen"
    assert decision["source"] == "environment"


def test_default_code_iterator_prunes_managed_directories_before_descent(
    monkeypatch,
    tmp_path,
):
    descended = []

    def fake_walk(root, *, topdown, followlinks):
        assert topdown is True
        assert followlinks is False
        dirnames = ["keep", "AI_Children", ".git", "venv", "node_modules"]
        yield root, dirnames, ["root.py"]
        descended.extend(dirnames)
        for dirname in dirnames:
            yield Path(root) / dirname, [], [f"{dirname}.py"]

    monkeypatch.setattr(rfm.os, "walk", fake_walk)

    yielded = [
        path.relative_to(tmp_path).as_posix()
        for path in rfm._iter_self_read_files(tmp_path, prune_generated=True)
    ]

    assert descended == ["keep"]
    assert yielded == ["root.py", "keep/keep.py"]


def test_default_code_iterator_never_yields_generated_tree_files(tmp_path):
    expected = {"root.py", "src/kept.py"}
    (tmp_path / "root.py").write_text("root", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("kept", encoding="utf-8")
    for dirname in rfm.DEFAULT_CODE_SCAN_PRUNED_DIRS:
        generated = tmp_path / dirname / "nested"
        generated.mkdir(parents=True)
        (generated / "ignored.py").write_text("ignored", encoding="utf-8")

    yielded = {
        path.relative_to(tmp_path).as_posix()
        for path in rfm._iter_self_read_files(tmp_path, prune_generated=True)
    }

    assert yielded == expected


def test_only_default_code_root_is_pruned_and_explicit_roots_remain_readable(tmp_path):
    default_root = tmp_path / "project"
    explicit_work = default_root / "AI_Children" / "Ina" / "work"
    github_history = default_root / "AI_Children" / "Ina" / "memory" / "github_history"
    explicit_work.mkdir(parents=True)
    github_history.mkdir(parents=True)
    (explicit_work / "song.py").write_text("work", encoding="utf-8")
    (github_history / "commit.txt").write_text("history", encoding="utf-8")

    assert rfm._should_prune_default_code_scan(default_root, default_root, "code")
    assert not rfm._should_prune_default_code_scan(explicit_work, default_root, "code")
    assert not rfm._should_prune_default_code_scan(
        github_history,
        default_root,
        "github_history",
    )
    assert [path.name for path in rfm._iter_self_read_files(explicit_work)] == [
        "song.py"
    ]
    assert [path.name for path in rfm._iter_self_read_files(github_history)] == [
        "commit.txt"
    ]


def test_malformed_history_fails_closed_without_overwriting(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    history_path = tmp_path / "AI_Children" / "Ina" / "memory" / "read_history.json"
    history_path.parent.mkdir(parents=True)
    original = b'{"version": 2, "files": '
    history_path.write_bytes(original)

    with pytest.raises(rfm.SelfReadHistoryLoadError):
        rfm.load_history("Ina")

    assert history_path.read_bytes() == original


def test_self_read_aborts_before_scan_when_history_is_unreadable(monkeypatch):
    messages = []

    monkeypatch.setattr(rfm, "get_child", lambda: "Ina")
    monkeypatch.setattr(
        rfm,
        "load_history",
        lambda child: (_ for _ in ()).throw(
            rfm.SelfReadHistoryLoadError("broken ledger")
        ),
    )
    monkeypatch.setattr(
        rfm,
        "load_self_read_preferences",
        lambda child: pytest.fail("preferences must not load after history failure"),
    )
    monkeypatch.setattr(
        rfm,
        "save_history",
        lambda *args, **kwargs: pytest.fail("history must not be overwritten"),
    )
    monkeypatch.setattr(
        rfm,
        "_iter_self_read_files",
        lambda *args, **kwargs: pytest.fail("scanner must not start"),
    )
    monkeypatch.setattr(rfm, "log_to_statusbox", messages.append)
    monkeypatch.setattr(rfm, "_SELF_READ_LOCK_HELD", False)

    assert rfm.self_read_and_train() is False
    assert any("failed closed" in message for message in messages)


def test_resume_offset_is_bound_to_the_file_fingerprint():
    stamp = {"mtime_ns": 100, "size_bytes": 20}
    prior = {
        **stamp,
        "continuation": {
            "offset": 2,
            "total_fragments": 5,
            "fingerprint": dict(stamp),
        },
    }

    assert rfm.classify_self_read_file(prior, stamp) == "resume"
    window, start, end, total = rfm._self_read_fragment_window(
        ["zero", "one", "two", "three", "four"],
        prior,
        stamp,
        2,
    )
    assert window == ["two", "three"]
    assert (start, end, total) == (2, 4, 5)

    changed = {"mtime_ns": 101, "size_bytes": 20}
    assert rfm.classify_self_read_file(prior, changed) == "updated"
    changed_window, changed_start, _, _ = rfm._self_read_fragment_window(
        ["zero", "one", "two"],
        prior,
        changed,
        2,
    )
    assert changed_start == 0
    assert changed_window == ["zero", "one"]


def _configure_bounded_self_read_pass(
    monkeypatch,
    tmp_path,
    *,
    paths,
    fragments_by_name,
    stamps_by_name,
    ledger,
    focus,
    fragment_limit,
):
    state = {
        "ledger": json.loads(json.dumps(ledger)),
        "passes": [],
        "focus": focus,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rfm, "FRAG_LIMIT", fragment_limit)
    monkeypatch.setattr(rfm, "get_child", lambda: "Ina")
    monkeypatch.setattr(rfm, "_SELF_READ_LOCK_HELD", False)
    monkeypatch.setattr(
        rfm,
        "load_history",
        lambda child: json.loads(json.dumps(state["ledger"])),
    )

    def capture_history(child, history):
        state["ledger"] = json.loads(json.dumps(history))
        state["passes"].append(json.loads(json.dumps(history["last_pass"])))

    monkeypatch.setattr(rfm, "save_history", capture_history)
    monkeypatch.setattr(
        rfm,
        "load_self_read_preferences",
        lambda child: {
            "source_choices": {
                "code": True,
                "books": False,
                "music": False,
                "venv": False,
                "github_history": False,
            },
            "skip_files": [],
        },
    )
    monkeypatch.setattr(rfm, "_apply_skip_requests", lambda child, prefs: prefs)
    monkeypatch.setattr(rfm, "_load_self_read_source_override", lambda: None)
    monkeypatch.setattr(
        rfm,
        "resolve_self_read_focus",
        lambda child: {
            "focus": state["focus"],
            "source": "test",
            "new_score": 0.5,
            "seen_score": 0.5,
            "drivers": {},
        },
    )
    monkeypatch.setattr(rfm, "book_folder_path", None)
    monkeypatch.setattr(rfm, "music_folder_path", None)
    monkeypatch.setattr(rfm, "ina_work_path", None)
    monkeypatch.setattr(rfm, "venv_path", None)
    monkeypatch.setattr(rfm, "FractalTransformer", lambda: object())
    monkeypatch.setattr(
        rfm,
        "_iter_self_read_files",
        lambda *args, **kwargs: iter(paths),
    )
    monkeypatch.setattr(rfm, "classify_path", lambda path: "text")
    monkeypatch.setattr(rfm, "is_readable_file", lambda path: True)
    monkeypatch.setattr(rfm, "_file_stamp", lambda path: dict(stamps_by_name[path.name]))
    monkeypatch.setattr(
        rfm,
        "fragment_text",
        lambda text, source, transformer, **_kwargs: json.loads(
            json.dumps(fragments_by_name[source])
        ),
    )
    monkeypatch.setattr(rfm, "log_reflection", lambda *args, **kwargs: None)
    monkeypatch.setattr(rfm, "log_to_statusbox", lambda *args, **kwargs: None)
    monkeypatch.setattr(rfm.os, "system", lambda command: 0)
    return state


def _test_fragments(prefix, count):
    return [
        {
            "id": f"{prefix}{index}",
            "tags": ["self_read"],
            "summary": f"{prefix}-{index}",
        }
        for index in range(count)
    ]


def test_truncated_file_resumes_tail_without_duplicate_head(monkeypatch, tmp_path):
    source = tmp_path / "long.py"
    source.write_text("long", encoding="utf-8")
    state = _configure_bounded_self_read_pass(
        monkeypatch,
        tmp_path,
        paths=[source],
        fragments_by_name={"long.py": _test_fragments("part", 5)},
        stamps_by_name={"long.py": {"mtime_ns": 10, "size_bytes": 4}},
        ledger=rfm._empty_read_history(),
        focus="new",
        fragment_limit=2,
    )

    expected_ranges = [(0, 2), (2, 4), (4, 5)]
    expected_ids = [["part0", "part1"], ["part2", "part3"], ["part4"]]
    for expected_range, fragment_ids in zip(expected_ranges, expected_ids):
        rfm.self_read_and_train()
        record = next(iter(state["ledger"]["files"].values()))
        assert (
            record["last_fragment_range"]["start"],
            record["last_fragment_range"]["end_exclusive"],
        ) == expected_range
        assert record["last_fragment_ids"] == fragment_ids

    final_record = next(iter(state["ledger"]["files"].values()))
    assert "continuation" not in final_record
    assert state["passes"][0]["read_reason_counts"]["new"] == 1
    assert state["passes"][1]["read_reason_counts"]["resume"] == 1
    assert state["passes"][2]["read_reason_counts"]["resume"] == 1


def test_seen_focus_reserves_fragment_for_due_revisit(monkeypatch, tmp_path):
    new_path = tmp_path / "new.py"
    seen_path = tmp_path / "seen.py"
    new_path.write_text("new", encoding="utf-8")
    seen_path.write_text("seen", encoding="utf-8")
    stamp = {"mtime_ns": 10, "size_bytes": 4}
    default_root = Path(rfm.__file__).resolve().parent
    seen_key = rfm.self_read_history_key("code", default_root, "seen.py")
    ledger = rfm._empty_read_history()
    ledger["files"][seen_key] = {
        "source": "code",
        "relative_path": "seen.py",
        "root_path": str(default_root),
        **stamp,
        "first_read_at": "2020-01-01T00:00:00+00:00",
        "last_read_at": "2020-01-01T00:00:00+00:00",
        "read_count": 1,
        "last_read_reason": "new",
    }
    state = _configure_bounded_self_read_pass(
        monkeypatch,
        tmp_path,
        paths=[new_path, seen_path],
        fragments_by_name={
            "new.py": _test_fragments("new", 4),
            "seen.py": _test_fragments("seen", 1),
        },
        stamps_by_name={"new.py": stamp, "seen.py": stamp},
        ledger=ledger,
        focus="seen",
        fragment_limit=3,
    )

    rfm.self_read_and_train()

    last_pass = state["passes"][-1]
    new_record = next(
        record
        for record in state["ledger"]["files"].values()
        if record.get("relative_path") == "new.py"
    )
    assert last_pass["fragments_saved"] == 3
    assert last_pass["seen_revisit_fragment_reserve"] == 1
    assert last_pass["seen_revisit_satisfied"] is True
    assert last_pass["read_reason_counts"]["new"] == 1
    assert last_pass["read_reason_counts"]["revisit"] == 1
    assert new_record["continuation"]["offset"] == 2
    assert (tmp_path / "AI_Children" / "Ina" / "memory" / "fragments" / "seen0.json").exists()
    assert not (tmp_path / "AI_Children" / "Ina" / "memory" / "fragments" / "new2.json").exists()



@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../Ina", "Ina/Other", r"Ina\Other", "/tmp/Ina", "Ina name"],
)
def test_child_identifier_rejects_path_escape(value):
    with pytest.raises(rfm.InvalidChildIdentifierError):
        rfm.validate_child_identifier(value)


def test_child_path_rejects_symlink_escape(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    managed = tmp_path / "AI_Children"
    outside = tmp_path / "outside"
    managed.mkdir()
    outside.mkdir()
    (managed / "Ina").symlink_to(outside, target_is_directory=True)

    with pytest.raises(rfm.InvalidChildIdentifierError):
        rfm._child_root_path("Ina")


def test_child_memory_path_rejects_relative_escape(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AI_Children" / "Ina" / "memory").mkdir(parents=True)

    with pytest.raises(rfm.InvalidChildIdentifierError):
        rfm._child_memory_path("Ina", "..", "outside.json")


def test_child_memory_path_rejects_memory_symlink_escape(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    child_root = tmp_path / "AI_Children" / "Ina"
    outside = tmp_path / "outside_memory"
    child_root.mkdir(parents=True)
    outside.mkdir()
    (child_root / "memory").symlink_to(outside, target_is_directory=True)

    with pytest.raises(rfm.InvalidChildIdentifierError):
        rfm._child_memory_path("Ina", "read_history.json")


def test_streaming_iterator_checks_deadline_across_empty_directories(
    monkeypatch,
    tmp_path,
):
    current = tmp_path
    for index in range(6):
        current = current / f"empty_{index}"
        current.mkdir()

    real_scandir = rfm.os.scandir
    opened = []
    checks = {"count": 0}

    def tracking_scandir(path):
        opened.append(Path(path))
        return real_scandir(path)

    def stop_requested():
        checks["count"] += 1
        return checks["count"] >= 5

    monkeypatch.setattr(rfm.os, "scandir", tracking_scandir)

    yielded = list(
        rfm._iter_self_read_files(
            tmp_path,
            stop_requested=stop_requested,
        )
    )

    assert yielded == []
    assert checks["count"] == 5
    assert len(opened) < 7


def test_streaming_iterator_prunes_generated_directories(tmp_path):
    keep = tmp_path / "keep"
    ignored = tmp_path / "AI_Children" / "nested"
    keep.mkdir()
    ignored.mkdir(parents=True)
    (keep / "kept.py").write_text("kept", encoding="utf-8")
    (ignored / "ignored.py").write_text("ignored", encoding="utf-8")

    yielded = {
        path.relative_to(tmp_path).as_posix()
        for path in rfm._iter_self_read_files(
            tmp_path,
            prune_generated=True,
            stop_requested=lambda: False,
        )
    }

    assert yielded == {"keep/kept.py"}


def test_get_child_ignores_pytest_flags_when_imported(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHILD", raising=False)
    monkeypatch.setattr(rfm.sys, "argv", ["pytest", "-q"])
    (tmp_path / "config.json").write_text(
        json.dumps({"current_child": "Ina_Test-2"}),
        encoding="utf-8",
    )

    assert rfm.get_child() == "Ina_Test-2"


def test_invalid_config_child_fails_before_child_writes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHILD", raising=False)
    monkeypatch.setattr(rfm.sys, "argv", ["pytest", "-q"])
    (tmp_path / "config.json").write_text(
        json.dumps({"current_child": "../Escape"}),
        encoding="utf-8",
    )

    with pytest.raises(rfm.InvalidChildIdentifierError):
        rfm.get_child()

    assert not (tmp_path / "Escape").exists()


@pytest.mark.parametrize(
    "record",
    [
        {
            "read_count": 1,
            "mtime_ns": "not-an-integer",
            "size_bytes": 4,
        },
        {
            "read_count": 1,
            "mtime_ns": -1,
            "size_bytes": 4,
        },
        {
            "read_count": 1,
            "mtime_ns": 10,
            "size_bytes": 4,
            "continuation": {
                "offset": 2,
                "total_fragments": 2,
                "fingerprint": {"mtime_ns": 10, "size_bytes": 4},
            },
        },
        {
            "read_count": 1,
            "mtime_ns": 10,
            "size_bytes": 4,
            "continuation": {
                "offset": 1,
                "total_fragments": 2,
                "fingerprint": {"mtime_ns": 11, "size_bytes": 4},
            },
        },
        {
            "read_count": 1,
            "mtime_ns": 10,
            "size_bytes": 4,
            "continuation": {
                "offset": 1,
                "total_fragments": 2,
                "fingerprint": {"mtime_ns": -1, "size_bytes": 4},
            },
        },
    ],
)
def test_v2_history_rejects_invalid_fingerprint_or_continuation(
    monkeypatch,
    tmp_path,
    record,
):
    monkeypatch.chdir(tmp_path)
    history_path = (
        tmp_path / "AI_Children" / "Ina" / "memory" / "read_history.json"
    )
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps({"version": 2, "files": {"entry": record}}),
        encoding="utf-8",
    )

    with pytest.raises(rfm.SelfReadHistoryLoadError):
        rfm.load_history("Ina")


def test_invalid_direct_fingerprint_is_treated_as_updated():
    assert (
        rfm.classify_self_read_file(
            {"mtime_ns": "bad", "size_bytes": 4},
            {"mtime_ns": 10, "size_bytes": 4},
        )
        == "updated"
    )



def test_music_iterator_accepts_audio_stem_archives_and_language_context(tmp_path):
    names = (
        "master.wav",
        "preview.mp3",
        "symbols.opus",
        "lyrics and style.txt",
        "notes.md",
        "manifest.json",
        "Godhunter Stems.zip",
        "cover.png",
        "canvas.mp4",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    expected = sorted(names, key=str.casefold)

    legacy = [
        path.name
        for path in rfm._iter_self_read_files(tmp_path, audio_only=True)
    ]
    streaming = sorted(
        (
            path.name
            for path in rfm._iter_self_read_files(
                tmp_path,
                audio_only=True,
                stop_requested=lambda: False,
            )
        ),
        key=str.casefold,
    )

    assert legacy == expected
    assert streaming == expected


def test_music_archive_filters_before_fragmenting_and_keeps_member_context(
    monkeypatch, tmp_path
):
    archive_path = tmp_path / "Godhunter's Lullaby Stems.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("stems/0 Lead Vocals.wav", b"audio")
        archive.writestr("lyrics and style.txt", b"words")
        archive.writestr("cover.png", b"image")

    calls = []

    def fragments(data, inner_path, container_path, category, _transformer, **_kwargs):
        calls.append((inner_path.as_posix(), category, data))
        return [
            {
                "modality": category,
                "source": f"{container_path.name}:{inner_path.as_posix()}",
                "tags": ["self_read"],
            }
        ]

    monkeypatch.setattr(rfm, "_fragments_from_data_buffer", fragments)

    result = rfm.process_archive(
        archive_path,
        object(),
        allowed_categories={"audio", "text", "image"},
    )

    assert [(name, category) for name, category, _data in calls] == [
        ("stems/0 Lead Vocals.wav", "audio"),
        ("lyrics and style.txt", "text"),
        ("cover.png", "image"),
    ]
    assert len(result) == 3
    assert result[0]["source_context"]["archive_container_name"] == archive_path.name
    assert result[0]["source_context"]["archive_member_path"] == "stems/0 Lead Vocals.wav"
    assert "archive_audio_member" in result[0]["tags"]
    assert "archive_text_member" in result[1]["tags"]

    for fragment in result:
        rfm.annotate_fragment_source(
            fragment,
            "music",
            "Ina Sings: Godhunter's Lullaby/Godhunter's Lullaby Stems.zip",
            tmp_path,
        )

    audio = result[0]
    assert "music_stem" in audio["tags"]
    assert audio["source_context"]["music_asset_kind"] == "stem"
    assert audio["source_context"]["stem_label"] == "0 Lead Vocals"
    assert audio["source_context"]["stem_container_relative_path"].endswith("Stems.zip")
    text = result[1]
    assert "music_language" in text["tags"]
    assert "lyrics_style_context" in text["tags"]
    assert "self_voice" not in text["tags"]
    assert text["source_context"]["ownership_hint"] == "self_creation"
    assert text["source_context"]["music_asset_kind"] == "lyrics_style_context"
    cover = result[2]
    assert "album_cover" in cover["tags"]
    assert cover["visual_learning"]["practice_use"] == "drawing"
    assert set(cover["visual_learning"]["alignment_keys"]) & set(audio["language_learning"]["alignment_keys"])


def test_direct_music_lyrics_use_language_not_audio_provenance(tmp_path):
    fragment = {"modality": "text", "tags": ["self_read"]}

    rfm.annotate_fragment_source(
        fragment,
        "music",
        "Ina Sings: Godhunter's Lullaby/lyrics and style.txt",
        tmp_path,
    )

    assert "ina_music" in fragment["tags"]
    assert "music_language" in fragment["tags"]
    assert "lyrics_style_context" in fragment["tags"]
    assert "self_voice" not in fragment["tags"]
    assert fragment["provenance"] == "ina_music_language_context"
    assert fragment["source_context"]["ownership_hint"] == "self_creation"


def test_generic_music_text_keeps_neutral_attribution(tmp_path):
    fragment = {"modality": "text", "tags": ["self_read"]}

    rfm.annotate_fragment_source(
        fragment,
        "music",
        "Uncatalogued Collection/notes.md",
        tmp_path,
    )

    assert "music_language" in fragment["tags"]
    assert "music_context" in fragment["tags"]
    assert "ina_music" not in fragment["tags"]
    assert "external_music" not in fragment["tags"]
    assert fragment["provenance"] == "music_language_context"
    assert fragment["source_context"]["ownership_hint"] == "unattributed"
    assert fragment["source_context"]["music_asset_kind"] == "music_context"


def test_studio_collection_assets_link_to_parent_and_manifest(tmp_path):
    studio_root = tmp_path / "AI_Children" / "Ina" / "memory" / "music_studio" / "stems"
    collection = "Godhunter_Lullaby_20260807"
    audio = {"modality": "audio", "tags": ["self_read"]}
    lyrics = {"modality": "text", "tags": ["self_read"]}

    rfm.annotate_fragment_source(
        audio,
        "music",
        f"{collection}/01_Lead_Vocals.wav",
        studio_root,
    )
    rfm.annotate_fragment_source(
        lyrics,
        "music",
        f"{collection}/context_01_lyrics_and_style.txt",
        studio_root,
    )

    for fragment in (audio, lyrics):
        context = fragment["source_context"]
        assert context["stem_collection_relative_path"] == collection
        assert context["stem_manifest_relative_path"] == f"{collection}/manifest.json"

    assert "music_stem" in audio["tags"]
    assert audio["source_context"]["music_asset_kind"] == "stem"
    assert audio["source_context"]["stem_label"] == "01_Lead_Vocals"
    assert "ina_music" in lyrics["tags"]
    assert lyrics["provenance"] == "ina_music_language_context"
    assert lyrics["source_context"]["ownership_hint"] == "self_creation"
    assert lyrics["source_context"]["stem_collection_context"] is True


def test_signed_external_music_retains_asset_classification(tmp_path):
    stem = {
        "modality": "audio",
        "tags": ["self_read"],
        "source_context": {
            "archive_member_path": "stems/Lead Guitar.wav",
            "archive_member_category": "audio",
        },
    }
    lyrics = {"modality": "text", "tags": ["self_read"]}
    relative_zip = "Rapidcrest: Sunshine Anthem/Sunshine Stems.zip"

    rfm.annotate_fragment_source(stem, "music", relative_zip, tmp_path)
    rfm.annotate_fragment_source(
        lyrics,
        "music",
        "Rapidcrest: Sunshine Anthem/lyrics and style.txt",
        tmp_path,
    )

    for fragment in (stem, lyrics):
        assert "external_music" in fragment["tags"]
        assert "signed_artist" in fragment["tags"]
        assert fragment["provenance"] == "signed_artist_catalog"
        assert fragment["source_context"]["ownership_hint"] == "external_artist"
        assert fragment["source_context"]["external_artist_hint"] == "rapidcrest"

    assert "music_stem" in stem["tags"]
    assert stem["source_context"]["stem_label"] == "Lead Guitar"
    assert "music_language" in lyrics["tags"]
    assert "lyrics_style_context" in lyrics["tags"]
    assert lyrics["source_context"]["music_asset_kind"] == "lyrics_style_context"


def test_archive_member_tag_uses_the_actual_category(monkeypatch, tmp_path):
    archive_path = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cover.png", b"image")

    monkeypatch.setattr(
        rfm,
        "_fragments_from_data_buffer",
        lambda *_args, **_kwargs: [{"modality": "image", "tags": []}],
    )

    fragments = rfm.process_archive(archive_path, object())

    assert len(fragments) == 1
    assert "archive_member" in fragments[0]["tags"]
    assert "archive_image_member" in fragments[0]["tags"]
    assert "archive_text_member" not in fragments[0]["tags"]


def test_child_studio_stems_are_scanned_before_external_music(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    child = "Ina"
    studio_root = (
        tmp_path / "AI_Children" / child / "memory" / "music_studio" / "stems"
    )
    external_root = tmp_path / "external_music"
    studio_root.mkdir(parents=True)
    external_root.mkdir()
    scanned = []

    monkeypatch.setattr(rfm, "get_child", lambda: child)
    monkeypatch.setattr(rfm, "load_history", lambda _child: rfm._empty_read_history())
    monkeypatch.setattr(
        rfm,
        "load_self_read_preferences",
        lambda _child: {
            "source_choices": {
                "code": False,
                "music": True,
                "books": False,
                "venv": False,
                "github_history": False,
            },
            "skip_files": [],
        },
    )
    monkeypatch.setattr(rfm, "_apply_skip_requests", lambda _child, prefs: prefs)
    monkeypatch.setattr(rfm, "_load_self_read_source_override", lambda: "music")
    monkeypatch.setattr(
        rfm,
        "resolve_self_read_focus",
        lambda _child: {
            "focus": "new",
            "source": "test",
            "new_score": 1.0,
            "seen_score": 0.0,
            "drivers": {},
        },
    )
    monkeypatch.setattr(rfm, "music_folder_path", external_root)
    monkeypatch.setattr(rfm, "book_folder_path", None)
    monkeypatch.setattr(rfm, "ina_work_path", None)
    monkeypatch.setattr(rfm, "venv_path", None)
    monkeypatch.setattr(rfm, "FractalTransformer", lambda: object())
    monkeypatch.setattr(rfm, "_SELF_READ_LOCK_HELD", False)
    monkeypatch.setattr(rfm, "save_history", lambda *_args: None)
    monkeypatch.setattr(rfm, "log_to_statusbox", lambda *_args: None)

    def empty_scan(base_root, **_kwargs):
        scanned.append(Path(base_root).resolve())
        return iter(())

    monkeypatch.setattr(rfm, "_iter_self_read_files", empty_scan)

    rfm.self_read_and_train()

    assert scanned == [studio_root.resolve(), external_root.resolve()]


def test_archive_member_aggregate_and_fragment_budgets(monkeypatch, tmp_path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index in range(6):
            archive.writestr(f"{index}.txt", b"data")

    calls = []

    def fake_fragments(data, inner_path, container_path, category, transformer, **_kwargs):
        calls.append(inner_path.name)
        return [{"summary": inner_path.name, "tags": []}]

    monkeypatch.setattr(rfm, "_fragments_from_data_buffer", fake_fragments)

    member_bounded = rfm.process_archive(
        archive_path,
        object(),
        member_limit=2,
        aggregate_limit=100,
        fragment_limit=10,
    )
    assert len(member_bounded) == 2
    assert len(calls) == 2

    calls.clear()
    aggregate_bounded = rfm.process_archive(
        archive_path,
        object(),
        member_limit=6,
        aggregate_limit=5,
        fragment_limit=10,
    )
    assert len(aggregate_bounded) == 1
    assert len(calls) == 1

    calls.clear()

    def three_fragments(data, inner_path, container_path, category, transformer, **_kwargs):
        calls.append(inner_path.name)
        return [{"summary": str(index), "tags": []} for index in range(3)]

    monkeypatch.setattr(rfm, "_fragments_from_data_buffer", three_fragments)
    fragment_bounded = rfm.process_archive(
        archive_path,
        object(),
        member_limit=6,
        aggregate_limit=100,
        fragment_limit=2,
    )
    assert len(fragment_bounded) == 2
    assert len(calls) == 1

    oversized_path = tmp_path / "too_many_entries.zip"
    with zipfile.ZipFile(oversized_path, "w") as archive:
        for index in range(rfm.ARCHIVE_MEMBER_COUNT_LIMIT + 1):
            archive.writestr(f"{index}.txt", b"")

    calls.clear()
    assert rfm.process_archive(oversized_path, object()) == []
    assert calls == []


def test_self_read_file_inspection_budget_is_independent_of_fragments(
    monkeypatch,
    tmp_path,
):
    paths = []
    fragments = {}
    stamps = {}
    for name in ("a.py", "b.py", "c.py"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths.append(path)
        fragments[name] = _test_fragments(name, 1)
        stamps[name] = {"mtime_ns": 10, "size_bytes": len(name)}

    state = _configure_bounded_self_read_pass(
        monkeypatch,
        tmp_path,
        paths=paths,
        fragments_by_name=fragments,
        stamps_by_name=stamps,
        ledger=rfm._empty_read_history(),
        focus="new",
        fragment_limit=10,
    )
    monkeypatch.setenv(rfm.SELF_READ_INSPECTION_LIMIT_ENV, "2")
    monkeypatch.setenv(rfm.SELF_READ_SCAN_SECONDS_ENV, "60")

    rfm.self_read_and_train()

    last_pass = state["passes"][-1]
    assert last_pass["files_inspected"] == 2
    assert last_pass["inspection_stop_reason"] == "file_budget"
    assert len(state["ledger"]["files"]) == 2


def test_self_read_time_budget_stops_before_next_file(monkeypatch, tmp_path):
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    stamp = {"mtime_ns": 10, "size_bytes": 1}
    state = _configure_bounded_self_read_pass(
        monkeypatch,
        tmp_path,
        paths=[first, second],
        fragments_by_name={
            "a.py": _test_fragments("a", 1),
            "b.py": _test_fragments("b", 1),
        },
        stamps_by_name={"a.py": stamp, "b.py": stamp},
        ledger=rfm._empty_read_history(),
        focus="new",
        fragment_limit=10,
    )
    clock = {"now": 0.0}

    def timed_files(*args, **kwargs):
        yield first
        clock["now"] = 2.0
        yield second

    monkeypatch.setattr(rfm, "_iter_self_read_files", timed_files)
    monkeypatch.setattr(rfm.time, "monotonic", lambda: clock["now"])
    monkeypatch.setenv(rfm.SELF_READ_INSPECTION_LIMIT_ENV, "10")
    monkeypatch.setenv(rfm.SELF_READ_SCAN_SECONDS_ENV, "1")

    rfm.self_read_and_train()

    last_pass = state["passes"][-1]
    assert last_pass["files_inspected"] == 1
    assert last_pass["inspection_stop_reason"] == "time_budget"
    assert len(state["ledger"]["files"]) == 1


def test_main_returns_nonzero_when_history_load_fails(monkeypatch):
    releases = []
    monkeypatch.setattr(rfm, "_acquire_runtime_lock", lambda: True)
    monkeypatch.setattr(rfm, "_install_runtime_signal_handlers", lambda: None)
    monkeypatch.setattr(rfm, "self_read_and_train", lambda: False)
    monkeypatch.setattr(
        rfm,
        "_release_runtime_lock",
        lambda status="exited", **kwargs: releases.append((status, kwargs)),
    )

    assert rfm.main() == 1
    assert releases[-1][0] == "failed"
