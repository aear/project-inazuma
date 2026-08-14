import pytest

from ina_desktop.files import VirtualFileSystem, configured_drives


def _filesystem(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    (books / "story.txt").write_text("once", encoding="utf-8")
    personal = tmp_path / "personal"
    drives = configured_drives(
        {"book_folder_path": str(books), "ina_hdd_writable_path": str(personal)},
        "Ina", project_root=tmp_path,
    )
    fs = VirtualFileSystem(drives)
    fs.ensure_writable_roots()
    return fs


def test_media_drives_are_read_only_and_personal_hdd_is_writable(tmp_path):
    fs = _filesystem(tmp_path)

    assert fs.read("books", "story.txt") == b"once"
    with pytest.raises(PermissionError, match="read-only"):
        fs.write("books", "notes.txt", "no")

    path = fs.write("ina_hdd", "Thoughts/first.txt", "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert path.stat().st_mode & 0o111 == 0
    assert fs.rename("ina_hdd", "Thoughts/first.txt", "revisited.txt").name == "revisited.txt"


def test_explorer_blocks_escape_symlinks_and_every_execution_request(tmp_path):
    fs = _filesystem(tmp_path)
    with pytest.raises(PermissionError, match="leaves"):
        fs.write("ina_hdd", "../escape.txt", "no")
    with pytest.raises(PermissionError, match="execution"):
        fs.execute("ina_hdd", "thought.py")

    outside = tmp_path / "outside"
    outside.mkdir()
    link = fs.drives["ina_hdd"].root / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(PermissionError):
        fs.write("ina_hdd", "outside-link/escape.txt", "no")


def test_drive_description_never_advertises_execution(tmp_path):
    fs = _filesystem(tmp_path)
    descriptions = fs.describe()

    assert {item["id"] for item in descriptions} >= {"books", "ina_hdd"}
    assert all(item["execution_allowed"] is False for item in descriptions)
    assert "write" not in next(item for item in descriptions if item["id"] == "books")["capabilities"]


def test_personal_drive_derives_from_configured_durable_hdd(tmp_path):
    drives = configured_drives(
        {"storage_layout": {"durable_mount": str(tmp_path / "durable")}},
        "Ina", project_root=tmp_path,
    )
    personal = next(drive for drive in drives if drive.id == "ina_hdd")
    assert personal.root == tmp_path / "durable" / "Ina Files" / "Ina"
    assert personal.writable is True
