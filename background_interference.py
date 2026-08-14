"""Bounded idle-vs-loaded benchmarks for human-visible background interference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


MAX_PHASE_SECONDS = 30.0
MAX_RAW_SAMPLES = 512
THREAD_SAMPLE_SECONDS = 0.25
NUMERICAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _latency_summary(values: list[float], *, method: str) -> dict[str, Any]:
    if not values:
        return {"available": False, "method": method, "samples": 0}
    return {
        "available": True,
        "method": method,
        "samples": len(values),
        "median_ms": round((_percentile(values, 0.5) or 0.0) * 1000.0, 4),
        "p95_ms": round((_percentile(values, 0.95) or 0.0) * 1000.0, 4),
        "max_ms": round(max(values) * 1000.0, 4),
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class LinuxSystemProbe:
    """Small `/proc` snapshots; no process payloads or memory stores are read."""

    def __init__(self, proc_root: Path | str = "/proc") -> None:
        self.root = Path(proc_root)

    def cpu(self) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        for line in _read_text(self.root / "stat").splitlines():
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
                continue
            try:
                values = [int(value) for value in fields[1:]]
            except ValueError:
                continue
            total = sum(values)
            idle = (values[3] if len(values) > 3 else 0) + (values[4] if len(values) > 4 else 0)
            result[fields[0]] = (total, idle)
        return result

    def context_switches(self) -> int:
        for line in _read_text(self.root / "stat").splitlines():
            if line.startswith("ctxt "):
                try:
                    return int(line.split()[1])
                except (IndexError, ValueError):
                    return 0
        return 0

    def vmstat(self) -> dict[str, int]:
        result: dict[str, int] = {}
        wanted = {"nr_dirty", "nr_writeback", "nr_dirtied", "nr_written"}
        for line in _read_text(self.root / "vmstat").splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] in wanted:
                try:
                    result[fields[0]] = int(fields[1])
                except ValueError:
                    pass
        return result

    def io_pressure_total_us(self) -> int:
        for line in _read_text(self.root / "pressure" / "io").splitlines():
            if not line.startswith("some "):
                continue
            for field in line.split():
                if field.startswith("total="):
                    try:
                        return int(field.split("=", 1)[1])
                    except ValueError:
                        return 0
        return 0

    def involuntary_context_switches(self) -> int:
        total = 0
        # Process count is naturally bounded by /proc; only tiny status files are read.
        for status_path in self.root.glob("[0-9]*/status"):
            for line in _read_text(status_path).splitlines():
                if line.startswith("nonvoluntary_ctxt_switches:"):
                    try:
                        total += int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                    break
        return total

    @staticmethod
    def _status_fields(path: Path) -> dict[str, str]:
        wanted = {"Name", "Pid", "PPid", "State", "Threads"}
        result: dict[str, str] = {}
        for line in _read_text(path).splitlines():
            key, separator, value = line.partition(":")
            if separator and key in wanted:
                result[key] = value.strip()
        return result

    def _process_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for status_path in self.root.glob("[0-9]*/status"):
            fields = self._status_fields(status_path)
            try:
                pid = int(fields.get("Pid", status_path.parent.name))
                ppid = int(fields.get("PPid", "0"))
                threads = int(fields.get("Threads", "1"))
            except ValueError:
                continue
            rows.append({
                "pid": pid,
                "ppid": ppid,
                "name": fields.get("Name", "unknown"),
                "state": (fields.get("State") or "?")[:1],
                "threads": max(0, threads),
            })
        return rows

    def system_threads(self) -> dict[str, Any]:
        """Summarise thread fan-out without retaining process arguments."""
        rows = self._process_rows()
        logical_cpus = max(1, len(self.cpu()) or int(os.cpu_count() or 1))
        total_threads = sum(int(row["threads"]) for row in rows)
        top = sorted(rows, key=lambda row: (-int(row["threads"]), int(row["pid"])))[:12]
        return {
            "logical_cpu_count": logical_cpus,
            "process_count": len(rows),
            "thread_count": total_threads,
            "threads_per_logical_cpu": round(total_threads / logical_cpus, 4),
            "runnable_process_count": sum(row["state"] == "R" for row in rows),
            "uninterruptible_process_count": sum(row["state"] == "D" for row in rows),
            "max_threads_per_process": max((int(row["threads"]) for row in rows), default=0),
            "top_threaded_processes": [
                {"pid": row["pid"], "name": row["name"], "threads": row["threads"]}
                for row in top
            ],
        }

    def target_threads(self, root_pid: int) -> dict[str, Any]:
        """Measure threads and runnable workers in one benchmark-owned process tree."""
        rows = self._process_rows()
        children: dict[int, list[int]] = {}
        by_pid = {int(row["pid"]): row for row in rows}
        for row in rows:
            children.setdefault(int(row["ppid"]), []).append(int(row["pid"]))
        target_pids: set[int] = set()
        pending = [int(root_pid)]
        while pending:
            pid = pending.pop()
            if pid in target_pids:
                continue
            target_pids.add(pid)
            pending.extend(children.get(pid, ()))

        state_counts: dict[str, int] = {}
        thread_count = 0
        for pid in target_pids:
            row = by_pid.get(pid)
            thread_count += int(row["threads"]) if row else 0
            for stat_path in (self.root / str(pid) / "task").glob("[0-9]*/stat"):
                text = _read_text(stat_path)
                try:
                    state = text.rsplit(")", 1)[1].split()[0]
                except (IndexError, AttributeError):
                    continue
                state_counts[state] = state_counts.get(state, 0) + 1
        logical_cpus = max(1, len(self.cpu()) or int(os.cpu_count() or 1))
        return {
            "root_pid": int(root_pid),
            "process_count": sum(pid in by_pid for pid in target_pids),
            "thread_count": thread_count,
            "threads_per_logical_cpu": round(thread_count / logical_cpus, 4),
            "runnable_thread_count": state_counts.get("R", 0),
            "uninterruptible_thread_count": state_counts.get("D", 0),
            "thread_states": state_counts,
        }


class PipeWireErrorProbe:
    """Read cumulative PipeWire node error counters from a bounded `pw-top` pass."""

    def __init__(self, command: Sequence[str] = ("pw-top", "-b", "-n", "2")) -> None:
        self.command = tuple(str(part) for part in command)

    def __call__(self) -> dict[str, int]:
        try:
            completed = subprocess.run(
                self.command, capture_output=True, text=True, check=False, timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        counters: dict[str, int] = {}
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) < 10 or fields[0] not in {"R", "S", "I", "C"}:
                continue
            try:
                node_id, errors = fields[1], int(fields[8])
            except (ValueError, IndexError):
                continue
            counters[node_id] = max(errors, counters.get(node_id, 0))
        return counters


@dataclass(frozen=True)
class InterferencePolicy:
    max_audio_errors_per_second: float = 0.0
    max_input_p95_ms: float = 16.7
    max_frame_p95_ms: float = 33.4
    max_context_switch_increase_ratio: float = 2.0
    max_involuntary_increase_ratio: float = 2.0
    max_io_stall_ms_per_second: float = 5.0
    max_core_busy_percent: float = 95.0
    max_task_threads_per_logical_cpu: float = 1.0


class BackgroundInterferenceBenchmark:
    """Compare bounded idle and loaded windows using latency-sensitive metrics.

    `input_probe` and `frame_probe` may attach a real virtual-input round trip and
    desktop capture/presentation probe. Scheduler wake-up delay is always retained
    as a dependency-free input-dispatch proxy instead of substituting total CPU.
    """

    def __init__(
        self,
        *,
        phase_seconds: float = 1.0,
        sample_interval_seconds: float = 0.02,
        system_probe: LinuxSystemProbe | None = None,
        audio_error_probe: Callable[[], Mapping[str, int]] | None = None,
        input_probe: Callable[[], Any] | None = None,
        frame_probe: Callable[[], Any] | None = None,
        policy: InterferencePolicy | None = None,
    ) -> None:
        self.phase_seconds = max(0.1, min(MAX_PHASE_SECONDS, float(phase_seconds)))
        self.sample_interval_seconds = max(0.002, min(1.0, float(sample_interval_seconds)))
        self.system = system_probe or LinuxSystemProbe()
        self.audio = audio_error_probe or PipeWireErrorProbe()
        self.input_probe = input_probe
        self.frame_probe = frame_probe
        self.policy = policy or InterferencePolicy()

    @staticmethod
    def _counter_delta(before: Mapping[str, int], after: Mapping[str, int]) -> int:
        return sum(max(0, int(after[key]) - int(before.get(key, after[key]))) for key in after)

    @staticmethod
    def _per_core(before: Mapping[str, tuple[int, int]], after: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
        percentages: dict[str, float] = {}
        for name, (end_total, end_idle) in after.items():
            start_total, start_idle = before.get(name, (end_total, end_idle))
            total_delta = max(0, end_total - start_total)
            idle_delta = max(0, end_idle - start_idle)
            if total_delta:
                percentages[name] = round(100.0 * max(0, total_delta - idle_delta) / total_delta, 3)
        busiest = max(percentages, key=percentages.get) if percentages else None
        return {
            "available": bool(percentages), "busiest_core": busiest,
            "max_busy_percent": percentages.get(busiest) if busiest else None,
            "saturated_core_count": sum(value >= 95.0 for value in percentages.values()),
            "per_core_busy_percent": percentages,
        }

    def _phase(self, name: str, *, task_pid: int | None = None) -> dict[str, Any]:
        audio_before = dict(self.audio() or {})
        cpu_before = self.system.cpu()
        ctxt_before = self.system.context_switches()
        involuntary_before = self.system.involuntary_context_switches()
        vm_before = self.system.vmstat()
        pressure_before = self.system.io_pressure_total_us()
        threads_before = self.system.system_threads()
        started = time.perf_counter()
        target = started
        wake_latencies: list[float] = []
        input_latencies: list[float] = []
        frame_latencies: list[float] = []
        raw_samples: list[dict[str, Any]] = []
        peak_dirty = int(vm_before.get("nr_dirty", 0))
        peak_writeback = int(vm_before.get("nr_writeback", 0))
        probe_errors = {"input": 0, "frame": 0}

        target_thread_peak: dict[str, Any] | None = None
        next_thread_sample = started
        while True:
            now = time.perf_counter()
            if now - started >= self.phase_seconds:
                break
            target += self.sample_interval_seconds
            time.sleep(max(0.0, target - time.perf_counter()))
            woke = time.perf_counter()
            wake_latency = max(0.0, woke - target)
            wake_latencies.append(wake_latency)

            input_latency = None
            if self.input_probe is not None:
                probe_started = time.perf_counter()
                try:
                    self.input_probe()
                except Exception:
                    probe_errors["input"] += 1
                else:
                    input_latency = time.perf_counter() - probe_started
                    input_latencies.append(input_latency)

            frame_latency = None
            if self.frame_probe is not None:
                probe_started = time.perf_counter()
                try:
                    self.frame_probe()
                except Exception:
                    probe_errors["frame"] += 1
                else:
                    frame_latency = time.perf_counter() - probe_started
                    frame_latencies.append(frame_latency)

            vm = self.system.vmstat()
            peak_dirty = max(peak_dirty, int(vm.get("nr_dirty", 0)))
            peak_writeback = max(peak_writeback, int(vm.get("nr_writeback", 0)))
            if task_pid is not None and woke >= next_thread_sample:
                snapshot = self.system.target_threads(task_pid)
                if target_thread_peak is None:
                    target_thread_peak = dict(snapshot)
                else:
                    for field in (
                        "process_count", "thread_count", "threads_per_logical_cpu",
                        "runnable_thread_count", "uninterruptible_thread_count",
                    ):
                        target_thread_peak[field] = max(
                            target_thread_peak.get(field, 0), snapshot.get(field, 0),
                        )
                    peak_states = target_thread_peak.setdefault("thread_states", {})
                    for state, count in snapshot.get("thread_states", {}).items():
                        peak_states[state] = max(peak_states.get(state, 0), count)
                next_thread_sample = woke + THREAD_SAMPLE_SECONDS
            if len(raw_samples) < MAX_RAW_SAMPLES:
                raw_samples.append({
                    "offset_ms": round((woke - started) * 1000.0, 4),
                    "scheduler_wakeup_ms": round(wake_latency * 1000.0, 4),
                    "input_probe_ms": round(input_latency * 1000.0, 4) if input_latency is not None else None,
                    "frame_probe_ms": round(frame_latency * 1000.0, 4) if frame_latency is not None else None,
                    "dirty_pages": int(vm.get("nr_dirty", 0)),
                    "writeback_pages": int(vm.get("nr_writeback", 0)),
                })

        elapsed = max(0.000001, time.perf_counter() - started)
        vm_after = self.system.vmstat()
        audio_after = dict(self.audio() or {})
        audio_delta = self._counter_delta(audio_before, audio_after)
        input_summary = _latency_summary(input_latencies or wake_latencies, method=(
            "provided_input_roundtrip" if input_latencies else "scheduler_wakeup_proxy"
        ))
        threads_after = self.system.system_threads()
        return {
            "phase": name, "elapsed_seconds": round(elapsed, 6), "raw_samples": raw_samples,
            "audio": {
                "available": bool(audio_before or audio_after), "error_delta": audio_delta,
                "errors_per_second": round(audio_delta / elapsed, 4),
                "start": audio_before, "end": audio_after,
            },
            "input_latency": input_summary,
            "desktop_frame_latency": _latency_summary(frame_latencies, method="provided_frame_probe"),
            "context_switches_per_second": round(
                max(0, self.system.context_switches() - ctxt_before) / elapsed, 3
            ),
            "involuntary_context_switches_per_second": round(
                max(0, self.system.involuntary_context_switches() - involuntary_before) / elapsed, 3
            ),
            "writeback_pressure": {
                "peak_dirty_pages": peak_dirty, "peak_writeback_pages": peak_writeback,
                "dirtied_pages_per_second": round(max(0, int(vm_after.get("nr_dirtied", 0)) - int(vm_before.get("nr_dirtied", 0))) / elapsed, 3),
                "written_pages_per_second": round(max(0, int(vm_after.get("nr_written", 0)) - int(vm_before.get("nr_written", 0))) / elapsed, 3),
                "io_stall_ms_per_second": round(max(0, self.system.io_pressure_total_us() - pressure_before) / 1000.0 / elapsed, 4),
            },
            "per_core": self._per_core(cpu_before, self.system.cpu()),
            "threads": {
                "method": "proc_status_and_task_state",
                "system_start": threads_before,
                "sampling_interval_seconds": THREAD_SAMPLE_SECONDS,
                "system_end": threads_after,
                "system_thread_delta": int(threads_after.get("thread_count", 0)) - int(threads_before.get("thread_count", 0)),
                "task_peak": target_thread_peak,
            },
            "probe_errors": probe_errors,
        }

    def _comparison(self, baseline: Mapping[str, Any], loaded: Mapping[str, Any]) -> dict[str, Any]:
        def value(container: Mapping[str, Any], *keys: str) -> float:
            current: Any = container
            for key in keys:
                current = current.get(key, {}) if isinstance(current, Mapping) else 0.0
            try:
                return float(current or 0.0)
            except (TypeError, ValueError):
                return 0.0

        ratios = {}
        for name, keys in {
            "context_switches": ("context_switches_per_second",),
            "involuntary_context_switches": ("involuntary_context_switches_per_second",),
        }.items():
            base, load = value(baseline, *keys), value(loaded, *keys)
            ratios[name] = round(load / max(1.0, base), 4)
        regressions = {
            "audio_xruns": value(loaded, "audio", "errors_per_second") > self.policy.max_audio_errors_per_second,
            "input_latency": value(loaded, "input_latency", "p95_ms") > self.policy.max_input_p95_ms,
            "desktop_frame_latency": (
                bool(loaded.get("desktop_frame_latency", {}).get("available"))
                and value(loaded, "desktop_frame_latency", "p95_ms") > self.policy.max_frame_p95_ms
            ),
            "context_switches": ratios["context_switches"] > self.policy.max_context_switch_increase_ratio,
            "involuntary_context_switches": ratios["involuntary_context_switches"] > self.policy.max_involuntary_increase_ratio,
            "writeback_pressure": value(loaded, "writeback_pressure", "io_stall_ms_per_second") > self.policy.max_io_stall_ms_per_second,
            "per_core_saturation": value(loaded, "per_core", "max_busy_percent") > self.policy.max_core_busy_percent,
            "thread_fanout": (
                bool(loaded.get("threads", {}).get("task_peak"))
                and value(loaded, "threads", "task_peak", "threads_per_logical_cpu")
                > self.policy.max_task_threads_per_logical_cpu
            ),
        }
        return {
            "loaded_minus_baseline": {
                "audio_errors_per_second": round(value(loaded, "audio", "errors_per_second") - value(baseline, "audio", "errors_per_second"), 4),
                "input_p95_ms": round(value(loaded, "input_latency", "p95_ms") - value(baseline, "input_latency", "p95_ms"), 4),
                "desktop_frame_p95_ms": round(value(loaded, "desktop_frame_latency", "p95_ms") - value(baseline, "desktop_frame_latency", "p95_ms"), 4),
                "context_switches_per_second": round(value(loaded, "context_switches_per_second") - value(baseline, "context_switches_per_second"), 3),
                "involuntary_context_switches_per_second": round(value(loaded, "involuntary_context_switches_per_second") - value(baseline, "involuntary_context_switches_per_second"), 3),
                "io_stall_ms_per_second": round(value(loaded, "writeback_pressure", "io_stall_ms_per_second") - value(baseline, "writeback_pressure", "io_stall_ms_per_second"), 4),
                "max_core_busy_percent": round(value(loaded, "per_core", "max_busy_percent") - value(baseline, "per_core", "max_busy_percent"), 3),
                "system_thread_count": round(value(loaded, "threads", "system_end", "thread_count") - value(baseline, "threads", "system_end", "thread_count"), 3),
            },
            "loaded_to_baseline_ratio": ratios,
            "regressions": regressions,
            "human_visible_regression": any(regressions.values()),
        }

    @staticmethod
    def _stop_owned_process(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    def run(
        self, command: Sequence[str], *, working_directory: Path | str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        command = tuple(str(part) for part in command)
        if not command:
            raise ValueError("a bounded background command is required")
        baseline = self._phase("baseline")
        env = dict(os.environ)
        if environment:
            env.update({str(key): str(value) for key, value in environment.items()})
        env["INA_BACKGROUND_INTERFERENCE_BENCHMARK"] = "1"
        process = subprocess.Popen(
            command, cwd=str(working_directory) if working_directory else None, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            loaded = self._phase("loaded", task_pid=process.pid)
        finally:
            self._stop_owned_process(process)
        return {
            "benchmark": "background_interference", "benchmark_version": "V1",
            "run_at": datetime.now(timezone.utc).isoformat(),
            "bounds": {
                "phase_seconds": self.phase_seconds,
                "sample_interval_seconds": self.sample_interval_seconds,
                "max_phase_seconds": MAX_PHASE_SECONDS,
                "max_raw_samples": MAX_RAW_SAMPLES,
            },
            "task": {
                "command": list(command), "returncode": process.returncode,
                "thread_environment": {name: env[name] for name in NUMERICAL_THREAD_VARIABLES if name in env},
                "numerical_thread_limits_explicit": any(name in env for name in NUMERICAL_THREAD_VARIABLES),
            },
            "baseline": baseline, "loaded": loaded,
            "comparison": self._comparison(baseline, loaded),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a bounded task after --")
    with tempfile.TemporaryDirectory(prefix="ina_background_interference_") as directory:
        result = BackgroundInterferenceBenchmark(
            phase_seconds=args.seconds, sample_interval_seconds=args.interval,
        ).run(command, working_directory=directory)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if result["comparison"]["human_visible_regression"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_PHASE_SECONDS", "MAX_RAW_SAMPLES", "NUMERICAL_THREAD_VARIABLES",
    "BackgroundInterferenceBenchmark",
    "InterferencePolicy", "LinuxSystemProbe", "PipeWireErrorProbe",
]
