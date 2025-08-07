import subprocess
import threading
from typing import Sequence, Union, Optional

from gui_hook import log_to_statusbox

Command = Union[str, Sequence[str]]


def safe_popen(command: Command, *, label: Optional[str] = None, verbose: bool = False,
               timeout: Optional[float] = None, **popen_kwargs) -> Optional[subprocess.Popen]:
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
            tag = f"ERROR" if is_err else (label or "LOG")
            for line in iter(stream.readline, ''):
                if line:
                    prefix = f"[{tag}] "
                    log_to_statusbox(prefix + line.rstrip() + "\n")
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
