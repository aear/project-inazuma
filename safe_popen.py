import subprocess
import threading
from pathlib import Path
from typing import Sequence, Union, Optional

from gui_hook import log_to_statusbox
from thread_governor import governed_environment

Command = Union[str, Sequence[str]]

_WAYLAND_ACTIVATION_WARNING = "Wayland does not support QWindow::requestActivate()"


def _stderr_tag(line: str) -> str:
    """Keep known compositor limitations from masquerading as app failures."""
    if _WAYLAND_ACTIVATION_WARNING in line:
        return "WARN"
    return "ERROR"


def safe_popen(command: Command, *, label: Optional[str] = None, verbose: bool = False,
               timeout: Optional[float] = None, governor_module: Optional[str] = None,
               governor_interactive: bool = False, **popen_kwargs) -> Optional[subprocess.Popen]:
    """Run subprocess.Popen with GUI-aware error handling.

    Parameters
    ----------
    command : list or str
        Command and arguments to execute.
    label : str, optional
        Label to prefix log lines with when verbose output is enabled.
    verbose : bool, default False
        If True, stream stdout and stderr to the GUI in real time.
    timeout : float, optional
        If set, kill the process if it runs longer than the given seconds.
    popen_kwargs : dict
        Additional keyword arguments forwarded to subprocess.Popen.
    """
    cmd_display = command if isinstance(command, str) else " ".join(map(str, command))
    if governor_module:
        popen_kwargs["env"] = governed_environment(
            governor_module,
            project_root=Path(__file__).resolve().parent,
            base=popen_kwargs.get("env"),
            workload="interactive" if governor_interactive else "background",
            interactive=governor_interactive,
        )
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if verbose else None,
            stderr=subprocess.PIPE if verbose else None,
            text=True,
            **popen_kwargs,
        )
    except Exception as exc:
        log_to_statusbox(f"[ERROR] Failed to launch '{cmd_display}': {exc}\n")
        return None

    if verbose:
        def _stream(stream, is_err=False):
            reported_once = set()
            for line in iter(stream.readline, ''):
                if line:
                    tag = _stderr_tag(line) if is_err else (label or "LOG")
                    normalized = line.rstrip()
                    if tag == "WARN" and normalized in reported_once:
                        continue
                    reported_once.add(normalized)
                    prefix = f"[{tag}] "
                    log_to_statusbox(prefix + normalized + "\n")
            stream.close()

        threading.Thread(target=_stream, args=(process.stdout, False), daemon=True).start()
        threading.Thread(target=_stream, args=(process.stderr, True), daemon=True).start()

    if timeout is not None:
        def _watch():
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                log_to_statusbox(
                    f"[ERROR] Command '{cmd_display}' timed out after {timeout} seconds\n"
                )
        threading.Thread(target=_watch, daemon=True).start()

    return process
