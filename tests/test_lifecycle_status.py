from lifecycle_status import format_lifecycle_status, read_lifecycle_status, update_lifecycle_status


def test_lifecycle_progress_is_bounded_and_safe_only_when_explicit(tmp_path):
    active = update_lifecycle_status(
        "Ina", operation="shutdown", phase="stopping_services",
        message="Stopping services", completed=2, total=4,
        remaining=["discord", "world"], root=tmp_path,
    )
    stopped = update_lifecycle_status(
        "Ina", operation="shutdown", phase="stopped",
        message="Complete", completed=9, total=4,
        safe_to_reboot=True, root=tmp_path,
    )

    assert active["progress"] == 0.5
    assert active["safe_to_reboot"] is False
    assert stopped["completed"] == 4
    assert stopped["progress"] == 1.0
    assert stopped["safe_to_reboot"] is True
    assert read_lifecycle_status("Ina", tmp_path)["phase"] == "stopped"


def test_lifecycle_display_reports_remaining_work_and_reboot_safety(tmp_path):
    status = update_lifecycle_status(
        "Ina", operation="shutdown", phase="flushing_storage",
        message="Flushing filesystem writes", completed=3, total=4,
        remaining=["storage flush"], safe_to_reboot=False, root=tmp_path,
    )
    rendered = format_lifecycle_status(status)

    assert "flushing storage" in rendered
    assert "remaining: storage flush" in rendered
    assert "safe to reboot" not in rendered
