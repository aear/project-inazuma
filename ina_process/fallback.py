"""Small psutil-compatible subset backed by Linux ``/proc``.

This is deliberately not a full psutil clone. It implements only Project
Inazuma's process ownership, lifecycle, CPU and memory reporting surface so a
broken package install does not disable containment or monitoring.
"""
from __future__ import annotations

import os
import signal
import time
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


class Error(Exception):
    pass


class NoSuchProcess(Error):
    pass


class AccessDenied(Error):
    pass


class ZombieProcess(Error):
    pass


STATUS_ZOMBIE = "zombie"

_MemoryInfo = namedtuple("pmem", "rss vms")
_MemoryFullInfo = namedtuple("pfullmem", "rss vms pss swap")
_CpuTimes = namedtuple("pcputimes", "user system")
_VirtualMemory = namedtuple("svmem", "total available percent used free")
_SwapMemory = namedtuple("sswap", "total used free percent")
_CLOCK_TICKS = max(1, int(os.sysconf("SC_CLK_TCK")))
_PAGE_SIZE = max(1, int(os.sysconf("SC_PAGE_SIZE")))
_SYSTEM_CPU_SAMPLE: Optional[tuple[int, int]] = None


def _proc_path(pid: int, name: str) -> Path:
    return Path("/proc") / str(pid) / name


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise NoSuchProcess(str(path)) from exc
    except PermissionError as exc:
        raise AccessDenied(str(path)) from exc
    except OSError as exc:
        raise Error(str(exc)) from exc


def _status_values(pid: int) -> dict[str, str]:
    values = {}
    for line in _read_text(_proc_path(pid, "status")).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key] = value.strip()
    return values


def _stat_fields(pid: int) -> tuple[str, list[str]]:
    raw = _read_text(_proc_path(pid, "stat")).strip()
    left = raw.find("(")
    right = raw.rfind(")")
    if left < 0 or right <= left:
        raise Error(f"Malformed /proc stat for pid {pid}")
    return raw[left + 1 : right], raw[right + 2 :].split()


def _meminfo() -> dict[str, int]:
    values = {}
    for line in _read_text(Path("/proc/meminfo")).splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        values[key] = value * 1024 if len(fields) > 1 and fields[1].lower() == "kb" else value
    return values


def _boot_time() -> float:
    try:
        uptime = float(_read_text(Path("/proc/uptime")).split()[0])
        return time.time() - uptime
    except (Error, ValueError, IndexError):
        return time.time()


class Process:
    def __init__(self, pid: Optional[int] = None) -> None:
        self.pid = int(os.getpid() if pid is None else pid)
        self.info: dict[str, object] = {}
        if self.pid <= 0 or not _proc_path(self.pid, "stat").exists():
            raise NoSuchProcess(str(self.pid))

    def __repr__(self) -> str:
        return f"Process(pid={self.pid})"

    def cmdline(self) -> list[str]:
        raw = _proc_path(self.pid, "cmdline")
        try:
            data = raw.read_bytes()
        except FileNotFoundError as exc:
            raise NoSuchProcess(str(self.pid)) from exc
        except PermissionError as exc:
            raise AccessDenied(str(self.pid)) from exc
        return [part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part]

    def cwd(self) -> str:
        try:
            return os.readlink(_proc_path(self.pid, "cwd"))
        except FileNotFoundError as exc:
            raise NoSuchProcess(str(self.pid)) from exc
        except PermissionError as exc:
            raise AccessDenied(str(self.pid)) from exc

    def name(self) -> str:
        name, _fields = _stat_fields(self.pid)
        return name

    def status(self) -> str:
        _name, fields = _stat_fields(self.pid)
        code = fields[0] if fields else "?"
        return {
            "R": "running", "S": "sleeping", "D": "disk-sleep",
            "T": "stopped", "t": "tracing-stop", "Z": STATUS_ZOMBIE,
            "X": "dead", "I": "idle",
        }.get(code, "unknown")

    def is_running(self) -> bool:
        try:
            return self.status() not in {"dead", STATUS_ZOMBIE}
        except NoSuchProcess:
            return False

    def create_time(self) -> float:
        _name, fields = _stat_fields(self.pid)
        try:
            return _boot_time() + (int(fields[19]) / _CLOCK_TICKS)
        except (IndexError, ValueError) as exc:
            raise Error(f"Malformed start time for pid {self.pid}") from exc

    def cpu_times(self):
        _name, fields = _stat_fields(self.pid)
        try:
            return _CpuTimes(int(fields[11]) / _CLOCK_TICKS, int(fields[12]) / _CLOCK_TICKS)
        except (IndexError, ValueError) as exc:
            raise Error(f"Malformed CPU fields for pid {self.pid}") from exc

    def memory_info(self):
        fields = _read_text(_proc_path(self.pid, "statm")).split()
        try:
            return _MemoryInfo(int(fields[1]) * _PAGE_SIZE, int(fields[0]) * _PAGE_SIZE)
        except (IndexError, ValueError) as exc:
            raise Error(f"Malformed memory fields for pid {self.pid}") from exc

    def memory_full_info(self):
        base = self.memory_info()
        pss = base.rss
        swap = 0
        path = _proc_path(self.pid, "smaps_rollup")
        try:
            for line in path.read_text(encoding="ascii", errors="replace").splitlines():
                key, separator, value = line.partition(":")
                if not separator:
                    continue
                fields = value.split()
                if not fields:
                    continue
                if key == "Pss":
                    pss = int(fields[0]) * 1024
                elif key == "Swap":
                    swap = int(fields[0]) * 1024
        except (OSError, ValueError):
            pass
        return _MemoryFullInfo(base.rss, base.vms, pss, swap)

    def num_threads(self) -> int:
        value = _status_values(self.pid).get("Threads", "1").split()[0]
        try:
            return int(value)
        except ValueError:
            return 1

    def children(self, recursive: bool = False) -> list["Process"]:
        by_parent: dict[int, list[Process]] = {}
        for candidate in process_iter(("pid",)):
            try:
                _name, fields = _stat_fields(candidate.pid)
                parent_pid = int(fields[1]) if len(fields) > 1 else 0
                by_parent.setdefault(parent_pid, []).append(candidate)
            except (Error, ValueError):
                continue
        direct = by_parent.get(self.pid, [])
        if not recursive:
            return direct
        found = []
        pending = list(direct)
        seen = {self.pid}
        while pending:
            child = pending.pop()
            if child.pid in seen:
                continue
            seen.add(child.pid)
            found.append(child)
            pending.extend(by_parent.get(child.pid, []))
        return found

    def terminate(self) -> None:
        self.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        self.send_signal(signal.SIGKILL)

    def send_signal(self, sig: int) -> None:
        try:
            os.kill(self.pid, sig)
        except ProcessLookupError as exc:
            raise NoSuchProcess(str(self.pid)) from exc
        except PermissionError as exc:
            raise AccessDenied(str(self.pid)) from exc

    def wait(self, timeout: Optional[float] = None):
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                waited, status = os.waitpid(self.pid, os.WNOHANG)
                if waited == self.pid:
                    return os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                pass
            if not _proc_path(self.pid, "stat").exists():
                return 0
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"process {self.pid} did not exit")
            time.sleep(0.02)


def process_iter(attrs: Optional[Sequence[str]] = None) -> Iterator[Process]:
    requested = tuple(attrs or ())
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            process = Process(int(entry.name))
            info = {}
            for attr in requested:
                if attr == "pid":
                    info[attr] = process.pid
                else:
                    method = getattr(process, attr)
                    info[attr] = method()
            process.info = info
            yield process
        except (Error, OSError):
            continue


def wait_procs(processes: Iterable[Process], timeout: Optional[float] = None):
    pending = list(processes)
    gone = []
    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    while pending:
        for process in list(pending):
            try:
                running = _proc_path(process.pid, "stat").exists() and process.is_running()
            except Error:
                running = False
            if not running:
                pending.remove(process)
                gone.append(process)
        if not pending or (deadline is not None and time.monotonic() >= deadline):
            break
        time.sleep(0.02)
    return gone, pending


def virtual_memory():
    values = _meminfo()
    total = int(values.get("MemTotal", 0))
    free = int(values.get("MemFree", 0))
    available = int(values.get("MemAvailable", free))
    used = max(0, total - available)
    percent = (used / total * 100.0) if total else 0.0
    return _VirtualMemory(total, available, percent, used, free)


def swap_memory():
    values = _meminfo()
    total = int(values.get("SwapTotal", 0))
    free = int(values.get("SwapFree", 0))
    used = max(0, total - free)
    percent = (used / total * 100.0) if total else 0.0
    return _SwapMemory(total, used, free, percent)


def _cpu_sample() -> tuple[int, int]:
    fields = _read_text(Path("/proc/stat")).splitlines()[0].split()[1:]
    ticks = [int(value) for value in fields]
    total = sum(ticks)
    idle = (ticks[3] if len(ticks) > 3 else 0) + (ticks[4] if len(ticks) > 4 else 0)
    return total, idle


def cpu_percent(interval: Optional[float] = None) -> float:
    global _SYSTEM_CPU_SAMPLE
    before = _cpu_sample()
    if interval is not None and float(interval) > 0:
        time.sleep(float(interval))
        previous = before
        current = _cpu_sample()
    else:
        previous = _SYSTEM_CPU_SAMPLE
        current = before
    _SYSTEM_CPU_SAMPLE = current
    if previous is None:
        return 0.0
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return 0.0
    return max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100.0))
