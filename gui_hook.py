import errno
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from self_read_reporting import report_self_read_broken_pipe

IS_WINDOWS = platform.system() == "Windows"
STATUS_PIPE_PATH = r"\\.\pipe\ina_status" if IS_WINDOWS else "/tmp/ina_status.pipe"
STATUS_LOG_PATH = Path(os.environ.get("INA_STATUS_LOG", "logs/ina_status.log"))
STATUS_PIPE_RETRY_DELAY_SECONDS = 30.0
STATUS_PIPE_REPORT_COOLDOWN_SECONDS = 180.0 * 60.0

_status_pipe_disabled_until = 0.0
_last_status_pipe_report_at = None


def _write_disk_log(message: str) -> None:
    try:
        STATUS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with STATUS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")
    except Exception:
        # Avoid recursive logging failures
        pass


def _is_self_read_message(message: str) -> bool:
    return "[SelfRead]" in str(message or "")


def _is_broken_pipe_error(error: BaseException) -> bool:
    return isinstance(error, BrokenPipeError) or "broken pipe" in str(error).lower()


def _is_expected_status_pipe_disconnect(error: BaseException) -> bool:
    if _is_broken_pipe_error(error):
        return True
    if isinstance(error, OSError):
        return error.errno in {errno.ENXIO, errno.ENOENT, errno.ENOTCONN}
    return False


def _suppress_status_pipe_temporarily() -> None:
    global _status_pipe_disabled_until
    _status_pipe_disabled_until = max(
        _status_pipe_disabled_until,
        time.monotonic() + STATUS_PIPE_RETRY_DELAY_SECONDS,
    )


def _status_pipe_suppressed() -> bool:
    return time.monotonic() < _status_pipe_disabled_until


def _report_self_read_status_pipe_issue(message: str, error: BaseException) -> bool:
    global _last_status_pipe_report_at
    text = str(message or "").strip()
    if not _is_self_read_message(text):
        return False

    now = time.monotonic()
    if (
        _last_status_pipe_report_at is not None
        and now - _last_status_pipe_report_at < STATUS_PIPE_REPORT_COOLDOWN_SECONDS
    ):
        return False
    _last_status_pipe_report_at = now

    report = report_self_read_broken_pipe(
        component="status_pipe",
        operation="status_log_write",
        error=error,
        source_message=text,
        path_text=STATUS_PIPE_PATH,
    )
    explanation = str(report.get("explanation") or "").strip()
    if not explanation:
        return False

    note = f"[SelfRead] Status pipe disconnected during status_log_write; scan is continuing without live status output. {explanation}"
    issue_entry_id = str(report.get("issue_entry_id") or "").strip()
    if issue_entry_id:
        note += f" GitHub queue entry: {issue_entry_id}."
    elif report.get("duplicate_within_cooldown"):
        note += " Existing cooldown report reused."
    _write_disk_log(note)
    print(note)
    return True


def _write_posix_status_pipe(message: str) -> None:
    flags = os.O_WRONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(STATUS_PIPE_PATH, flags)
    try:
        os.write(fd, f"{message}\n".encode("utf-8", errors="replace"))
    finally:
        os.close(fd)


def _write_windows_status_pipe(message: str) -> None:
    import pywin32_namedpipe as namedpipe  # hypothetical placeholder

    with namedpipe.NamedPipeClient(STATUS_PIPE_PATH) as pipe:
        pipe.write(message + "\n")


def _handle_status_pipe_failure(message: str, error: BaseException) -> bool:
    expected_disconnect = _is_expected_status_pipe_disconnect(error)
    if expected_disconnect:
        _suppress_status_pipe_temporarily()
        if _is_broken_pipe_error(error) and _report_self_read_status_pipe_issue(message, error):
            return True
        if not _is_self_read_message(message):
            print("[LogHook] Status pipe reader unavailable; using fallback logging.")
        return True

    print(f"[LogHook] Failed to write to status pipe: {error}")
    return False


def log_to_statusbox(message: str):
    _write_disk_log(message)
    print(message)

    if _status_pipe_suppressed():
        fallback_log(message, announce=False)
        return

    try:
        if IS_WINDOWS:
            _write_windows_status_pipe(message)
        elif os.path.exists(STATUS_PIPE_PATH):
            _write_posix_status_pipe(message)
        else:
            fallback_log(message, announce=False)
            return
    except Exception as e:
        handled = _handle_status_pipe_failure(message, e)
        fallback_log(message, announce=not handled)


def fallback_log(message: str, *, announce: bool = True):
    """
    Fallback logging to stdout or a file if the pipe is unavailable.
    """
    if announce:
        print(f"[LogHook] Logging to fallback method: {message}")
    _write_disk_log(message)
    try:
        with open("/tmp/ina_status_fallback.log", "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[LogHook] Failed to log to fallback file: {e}")
