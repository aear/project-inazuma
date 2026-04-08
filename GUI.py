import tkinter as tk
from tkinter import Menu, messagebox, filedialog, simpledialog
import json
import os
import sys
from datetime import datetime, timezone
from safe_popen import safe_popen
import psutil
import shutil
from pathlib import Path
from model_manager import get_inastate, update_inastate, request_meal, offer_meal
import threading
import time
from memory_graph import build_fractal_memory
import platform
from birth_system import boot
from emotion_engine import SLIDERS as EMOTION_SLIDERS, load_baseline
from emotion_processor import process_emotion
from collections import deque

STATUS_RETENTION_SEC = float(os.environ.get("INA_STATUS_RETENTION_SEC", "600"))
_status_buffer = deque()
_status_buffer_lock = threading.Lock()

def _status_line_count(msg: str) -> int:
    if not msg:
        return 0
    return msg.count("\n") + (0 if msg.endswith("\n") else 1)

def _record_status_entry(msg: str) -> int:
    now = time.time()
    line_count = _status_line_count(msg)
    removed_lines = 0
    with _status_buffer_lock:
        _status_buffer.append((now, line_count))
        cutoff = now - STATUS_RETENTION_SEC
        while _status_buffer and _status_buffer[0][0] < cutoff:
            removed_lines += _status_buffer.popleft()[1]
    return removed_lines

def append_status(msg, tag=None):
    """Safely append to the status box from any thread."""
    removed_lines = _record_status_entry(msg)
    def _append():
        if removed_lines > 0:
            try:
                status_box.delete("1.0", f"{removed_lines + 1}.0")
            except Exception:
                pass
        if tag:
            status_box.insert(tk.END, msg, tag)
        else:
            status_box.insert(tk.END, msg)
        status_box.see(tk.END)
    # Guard against early calls before the UI exists
    if "root" in globals():
        root.after(0, _append)


# === Pipe config for cross-module logging ===
IS_WINDOWS = platform.system() == "Windows"
STATUS_PIPE_PATH = r"\\.\pipe\ina_status" if IS_WINDOWS else "/tmp/ina_status.pipe"

def status_log_server():
    def run_pipe():
        if not IS_WINDOWS and os.path.exists(STATUS_PIPE_PATH):
            os.remove(STATUS_PIPE_PATH)
        if not IS_WINDOWS:
            os.mkfifo(STATUS_PIPE_PATH)

        while True:
            try:
                if IS_WINDOWS:
                    import pywin32_namedpipe as namedpipe  # hypothetical placeholder
                    with namedpipe.NamedPipeClient(STATUS_PIPE_PATH) as pipe:
                        while True:
                            msg = pipe.readline()
                            if msg:
                                tag = "error" if msg.startswith("[ERROR]") else None
                                append_status(msg, tag)
                else:
                    with open(STATUS_PIPE_PATH, "r") as pipe:
                        for msg in pipe:
                            if msg.strip():
                                tag = "error" if msg.startswith("[ERROR]") else None
                                append_status(msg, tag)
            except Exception as e:
                append_status(f"[Pipe Error] {e}\n")
                time.sleep(2)

    threading.Thread(target=run_pipe, daemon=True).start()

def clear_status_log():
    status_box.delete("1.0", tk.END)
    status_box.insert(tk.END, "[Log] Cleared status log.\n")
    status_box.see(tk.END)
    with _status_buffer_lock:
        _status_buffer.clear()

    # Purge all __pycache__ directories
    root = Path(".").resolve()
    pycaches = list(root.rglob("__pycache__"))
    for cache_dir in pycaches:
        try:
            shutil.rmtree(cache_dir)
            status_box.insert(tk.END, f"[Log] Removed: {cache_dir}\n")
            status_box.see(tk.END)
        except Exception as e:
            status_box.insert(tk.END, f"[Log] Failed to remove {cache_dir}: {e}\n")
            status_box.see(tk.END)

    status_box.insert(tk.END, "[Log] __pycache__ directories purged.\n")
    status_box.see(tk.END)




def stream_subprocess_to_status(command, label="Process"):
    def stream_output():
        append_status(f"[{label}] Starting...\n")
        process = safe_popen(command, label=label, verbose=True)
        if process is not None:
            process.wait()
            append_status(f"[{label}] Completed.\n")
        else:
            append_status(f"[{label}] Failed to start.\n", "error")

    threading.Thread(target=stream_output, daemon=True).start()


def signal_memory_too_high(source="gui", note=None):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "too_much_memory",
        "source": source,
        "note": note or "operator requested memory shed",
    }
    update_inastate("operator_memory_signal", payload)
    append_status("[Operator] Memory pressure signal queued.\n")


def _shortcut_memory_too_high(event=None):
    signal_memory_too_high(source="gui_shortcut", note="Ctrl+Shift+M")



CONFIG_FILE = "config.json"
CONFIG_DEFAULTS = {
    "book_folder_path": "",
    "music_folder_path": "",
}
config = dict(CONFIG_DEFAULTS)
book_path_var = None
music_path_var = None
model_running = False
vitals_window = None
_usage_labels = {}
energy_var = None
energy_status_var = None
emotion_vars = {}
hunger_status_var = None
fitness_status_var = None
nutrition_info_var = None
last_meal_status_var = None
metabolic_status_var = None
offer_status_var = None
offer_note_var = None
_last_resource_publish_ts = 0.0
_last_resource_publish_key = None
_process_cpu_samples = {}
RESOURCE_PUBLISH_INTERVAL_SEC = float(os.environ.get("INA_RESOURCE_PUBLISH_INTERVAL_SEC", "15"))
RESOURCE_HISTORY_MAX_SAMPLES = max(12, int(float(os.environ.get("INA_RESOURCE_HISTORY_MAX_SAMPLES", "240"))))
RESOURCE_TREND_SHORT_SAMPLES = max(4, int(float(os.environ.get("INA_RESOURCE_TREND_SHORT_SAMPLES", "8"))))
RESOURCE_TREND_LONG_SAMPLES = max(8, int(float(os.environ.get("INA_RESOURCE_TREND_LONG_SAMPLES", "40"))))


def refresh_config():
    global config
    data = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    config = dict(CONFIG_DEFAULTS)
    if isinstance(data, dict):
        config.update(data)

    if book_path_var is not None:
        book_path_var.set(config.get("book_folder_path", ""))
    if music_path_var is not None:
        music_path_var.set(config.get("music_folder_path", ""))


def save_config():
    global config
    config_path = CONFIG_FILE
    updated = dict(config)

    if 'root' in globals():
        updated["geometry"] = root.winfo_geometry()

    if book_path_var is not None:
        updated["book_folder_path"] = book_path_var.get()
    if music_path_var is not None:
        updated["music_folder_path"] = music_path_var.get()

    current = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                current = json.load(f)
        except json.JSONDecodeError:
            current = {}

    if isinstance(current, dict):
        current.update(updated)
    else:
        current = updated

    config = current

    with open(config_path, "w") as f:
        json.dump(current, f, indent=4)


def _update_folder_setting(key, var, description):
    if var is None:
        return

    new_value = var.get().strip()
    if config.get(key, "") == new_value:
        return

    var.set(new_value)
    config[key] = new_value
    status_box.insert(tk.END, f"[Config] {description} set to: {new_value or '(empty)'}\n")
    status_box.see(tk.END)
    save_config()


def commit_book_folder(event=None):
    _update_folder_setting("book_folder_path", book_path_var, "Book folder")
    if event and getattr(event, "keysym", None) == "Return":
        return "break"


def commit_music_folder(event=None):
    _update_folder_setting("music_folder_path", music_path_var, "Music folder")
    if event and getattr(event, "keysym", None) == "Return":
        return "break"


def browse_book_folder():
    if book_path_var is None:
        return
    initial_dir = book_path_var.get() or os.getcwd()
    path = filedialog.askdirectory(initialdir=initial_dir)
    if path:
        book_path_var.set(path)
        commit_book_folder()


def browse_music_folder():
    if music_path_var is None:
        return
    initial_dir = music_path_var.get() or os.getcwd()
    path = filedialog.askdirectory(initialdir=initial_dir)
    if path:
        music_path_var.set(path)
        commit_music_folder()

def birth_new_model():
    status_box.insert(tk.END, "Opening Birth Certificate window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "birth_certificate.py"], verbose=True)

def load_child():
    status_box.insert(tk.END, "Load Child selected.\n")
    status_box.see(tk.END)

    path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if not path:
        return

    try:
        with open(path, "r") as f:
            birth_data = json.load(f)

        name = f"{birth_data['given_name']}_{birth_data['family_name']}".strip()
        ai_dir = Path("AI_Children") / name
        memory_dir = ai_dir / "memory"

        if not ai_dir.exists():
            ai_dir.mkdir(parents=True)
            memory_dir.mkdir(parents=True)

            for file_name in ["memory.json", "memory_index.json", "memory_graph.json"]:
                src = Path(file_name)
                if src.exists():
                    shutil.move(str(src), memory_dir / file_name)

            frag_dir = Path("fragments")
            if frag_dir.exists():
                shutil.move(str(frag_dir), memory_dir / "fragments")

            shutil.copy(path, ai_dir / "birth_certificate.json")
            status_box.insert(tk.END, f"Organized new child: {name}\n")
        else:
            status_box.insert(tk.END, f"{name} is already organized.\n")

        config["current_child"] = name
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)

    except Exception as e:
        messagebox.showerror("Load Child Failed", f"Could not load child: {e}")
        status_box.insert(tk.END, f"[ERROR] {e}\n")

def save_load_config():
    status_box.insert(tk.END, "Save/Load Config selected.\n")
    status_box.see(tk.END)

def exceptions_list():
    status_box.insert(tk.END, "Opening Exceptions List window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "exception_window.py"], verbose=True)


def precision_settings():
    status_box.insert(tk.END, "Opening Precision Settings window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "precision_window.py"], verbose=True)

def open_timers_config():
    status_box.insert(tk.END, "Opening Timers configuration.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "timers_window.py"], verbose=True)

def open_audio_devices_window():
    status_box.insert(tk.END, "Opening Audio Devices window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "audio_device_window.py"], verbose=True)


def pretrain_mode():
    append_status("Entering Pretrain mode...\n")

    def stream_pretrain():
        # Fetch child from the current configuration
        config = load_config()
        child = config.get("current_child", "Inazuma_Yagami")

        append_status(f"[Pretrain] Using child: {child}\n")

        process = safe_popen([sys.executable, "pretrain_logic.py", child], label="Pretrain", verbose=True)
        if process is not None:
            process.wait()
            append_status("[Pretrain] Finished pretraining.\n")
        else:
            append_status("[Pretrain] Failed to start pretraining.\n", "error")
            
    threading.Thread(target=stream_pretrain, daemon=True).start()



def open_eeg_view():
    status_box.insert(tk.END, "Opening EEG window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "EEG.py"], label="EEG", verbose=True)

def update_ai_count_label():
    ai_count = 1 if model_running else 0
    canvas.itemconfig(ai_text_id, text=str(ai_count))

def start_model():
    global model_running
    if model_running:
        append_status("[GUI] Model already running.\n")
        return

    append_status("Start Button clicked.\n")
    append_status("Launching Birth System...\n")
    child = config.get("current_child", "default_child")

    def _boot():
        global model_running
        try:
            boot(child)
            model_running = True
            append_status("[GUI] Birth sequence returned.\n")
        except Exception as exc:
            model_running = False
            append_status(f"[GUI ERROR] Birth sequence failed: {exc}\n", tag="error")
        update_ai_count_label()

    threading.Thread(target=_boot, daemon=True).start()
    model_running = True
    update_ai_count_label()


def load_config():
    path = Path("config.json")
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def _clamp_value(value, lo=-1.0, hi=1.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def _ina_processes():
    try:
        root_proc = psutil.Process(os.getpid())
    except psutil.Error:
        return []
    try:
        children = root_proc.children(recursive=True)
    except psutil.Error:
        children = []
    return [root_proc] + children

def _process_cpu_sample_key(proc):
    try:
        created = round(float(proc.create_time()), 3)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        created = None
    return (int(proc.pid), created)


def _sample_process_cpu_percent(proc, now=None):
    now = time.monotonic() if now is None else float(now)
    key = _process_cpu_sample_key(proc)
    try:
        times = proc.cpu_times()
        total_cpu_time = float(getattr(times, 'user', 0.0)) + float(getattr(times, 'system', 0.0))
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return 0.0
    previous = _process_cpu_samples.get(key)
    _process_cpu_samples[key] = (now, total_cpu_time)
    if not previous:
        return 0.0
    elapsed = now - float(previous[0])
    cpu_delta = total_cpu_time - float(previous[1])
    if elapsed <= 0.0 or cpu_delta < 0.0:
        return 0.0
    return max(0.0, (cpu_delta / elapsed) * 100.0)


def _prune_process_cpu_samples(live_keys):
    for key in list(_process_cpu_samples):
        if key not in live_keys:
            _process_cpu_samples.pop(key, None)


def _prime_usage_counters():
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass
    now = time.monotonic()
    live_keys = set()
    for proc in _ina_processes():
        key = _process_cpu_sample_key(proc)
        live_keys.add(key)
        _sample_process_cpu_percent(proc, now=now)
    _prune_process_cpu_samples(live_keys)


def _module_process_name(proc):
    try:
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        cmdline = []
    for part in cmdline[1:]:
        name = os.path.basename(str(part))
        if name.endswith('.py'):
            return name
    try:
        return proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return f'pid:{proc.pid}'


def _format_ram_value(num_bytes):
    if not num_bytes:
        return '0 MB'
    units = (
        (1024 ** 3, 'GB'),
        (1024 ** 2, 'MB'),
        (1024, 'KB'),
    )
    for scale, suffix in units:
        if num_bytes >= scale:
            value = num_bytes / scale
            if suffix == 'GB':
                return f'{value:.2f} {suffix}'
            return f'{value:.1f} {suffix}'
    return f'{int(num_bytes)} B'


def _format_module_usage(modules):
    if not modules:
        return 'Per-process modules: none detected.'
    header = f"{'Module':<24} {'RAM':>10} {'CPU':>7} {'Thr':>5} {'Proc':>5}"
    lines = [header, '-' * len(header)]
    for item in modules:
        lines.append(
            f"{item['name'][:24]:<24} {_format_ram_value(item['mem_bytes']):>10} {item['cpu']:>6.1f}% {item['threads']:>5} {item['processes']:>5}"
        )
    return '\n'.join(lines)


def _format_process_usage(process_rows):
    if not process_rows:
        return 'Per-process view: none detected.'
    top_cpu = max(process_rows, key=lambda item: (float(item.get('cpu') or 0.0), int(item.get('mem_bytes') or 0), str(item.get('name') or '')))
    top_cpu_value = float(top_cpu.get('cpu') or 0.0)
    if top_cpu_value > 0.0:
        top_line = f"Top CPU: {top_cpu['name']} pid={int(top_cpu['pid'])} CPU {top_cpu_value:.1f}% RAM {_format_ram_value(int(top_cpu.get('mem_bytes') or 0))}"
    else:
        top_line = 'Top CPU: waiting for the next CPU sample.'
    header = f"{'PID':>7} {'Module':<24} {'RAM':>10} {'CPU':>7} {'Thr':>5}"
    lines = [top_line, header, '-' * len(header)]
    visible = list(process_rows[:24])
    for item in visible:
        lines.append(
            f"{int(item['pid']):>7} {item['name'][:24]:<24} {_format_ram_value(item['mem_bytes']):>10} {item['cpu']:>6.1f}% {item['threads']:>5}"
        )
    if len(process_rows) > len(visible):
        lines.append(f"...and {len(process_rows) - len(visible)} more process(es)")
    return '\n'.join(lines)


def _scheduler_snapshot():
    raw = get_inastate('process_scheduler') or {}
    if not isinstance(raw, dict):
        return {'available': False, 'summary': 'Scheduler data is not available yet.', 'learning_hint': '', 'running': [], 'next_slots': [], 'last_decisions': [], 'recent_activity': [], 'module_history': []}
    planner = raw.get('planner') if isinstance(raw.get('planner'), dict) else {}
    slot_summary = raw.get('slot_summary') if isinstance(raw.get('slot_summary'), dict) else {}
    return {
        'available': bool(raw),
        'summary': str(planner.get('summary') or 'Scheduler idle.').strip(),
        'learning_hint': str(planner.get('learning_hint') or '').strip(),
        'running': [item for item in (planner.get('running') or [])[:4] if isinstance(item, dict)],
        'next_slots': [item for item in (planner.get('next_slots') or [])[:10] if isinstance(item, dict)],
        'last_decisions': [item for item in (planner.get('last_decisions') or [])[-4:] if isinstance(item, dict)],
        'recent_activity': [item for item in (planner.get('recent_activity') or [])[:12] if isinstance(item, dict)],
        'module_history': [item for item in (planner.get('module_history') or [])[:8] if isinstance(item, dict)],
        'history_window_hours': round(float(planner.get('history_window_hours') or 24.0), 2),
        'cancelled_count': int(planner.get('cancelled_count') or 0),
        'queue_depth': int(planner.get('queue_depth') or slot_summary.get('queued_slots') or 0),
        'running_count': int(planner.get('running_count') or slot_summary.get('running_slots') or 0),
        'blocked_count': int(planner.get('blocked_count') or 0),
        'memory_guard_level': str(planner.get('memory_guard_level') or 'unknown').strip().lower() or 'unknown',
        'cpu_percent': round(float(planner.get('cpu_percent') or 0.0), 1),
        'gpu_utilization_percent': round(float(planner.get('gpu_utilization_percent') or 0.0), 1),
        'gpu_available': bool(planner.get('gpu_available', False)),
        'max_parallel_tasks': int(slot_summary.get('max_parallel_tasks') or 0),
        'max_queue_slots': int(slot_summary.get('max_queue_slots') or 0),
    }


def _summarize_scheduler_state(scheduler):
    if not scheduler.get('available'):
        return 'Scheduler data is not available yet.'
    summary = str(scheduler.get('summary') or 'Scheduler idle.').strip()
    hint = str(scheduler.get('learning_hint') or '').strip()
    if hint:
        return f'{summary} {hint}'
    return summary


def _format_scheduler_slots(scheduler):
    if not scheduler.get('available'):
        return 'Scheduler queue: no data yet.'
    lines = [
        f"Running {int(scheduler.get('running_count') or 0)}/{int(scheduler.get('max_parallel_tasks') or 0)}  |  Queue {int(scheduler.get('queue_depth') or 0)}/{int(scheduler.get('max_queue_slots') or 0)}  |  Guard {scheduler.get('memory_guard_level')}",
        f"CPU {float(scheduler.get('cpu_percent') or 0.0):.1f}%" + (f"  |  GPU {float(scheduler.get('gpu_utilization_percent') or 0.0):.1f}%" if scheduler.get('gpu_available') else ''),
    ]
    running = scheduler.get('running') or []
    queued = scheduler.get('next_slots') or []
    if running:
        lines.append('Running now:')
        for item in running[:4]:
            pid = item.get('pid')
            pid_text = f" pid={int(pid)}" if pid else ''
            cpu = item.get('cpu_percent')
            cpu_text = f" cpu={float(cpu):.1f}%" if cpu is not None else ''
            lines.append(f"- {item.get('label') or item.get('task_key')} (p{int(item.get('priority') or 0)}){pid_text}{cpu_text}")
    else:
        lines.append('Running now: idle')
    if queued:
        lines.append('Next slots:')
        for index, item in enumerate(queued[:10], 1):
            lines.append(f"{index:>2}. {item.get('label') or item.get('task_key')} (p{int(item.get('priority') or 0)})")
    else:
        lines.append('Next slots: empty')
    decisions = scheduler.get('last_decisions') or []
    blocked = [item for item in decisions if str(item.get('decision') or '').strip().lower() == 'blocked']
    if blocked:
        last = blocked[-1]
        lines.append(f"Last block: {last.get('label') or last.get('task_key')} ({str(last.get('reason') or '').replace('_', ' ')})")
    module_history = scheduler.get('module_history') or []
    if module_history:
        lines.append(f"24h module history ({float(scheduler.get('history_window_hours') or 24.0):.0f}h):")
        for item in module_history[:5]:
            counts = []
            if int(item.get('queued_count') or 0):
                counts.append(f"q{int(item.get('queued_count') or 0)}")
            if int(item.get('started_count') or 0):
                counts.append(f"run{int(item.get('started_count') or 0)}")
            if int(item.get('completed_count') or 0):
                counts.append(f"done{int(item.get('completed_count') or 0)}")
            if int(item.get('cancelled_count') or 0):
                counts.append(f"cancel{int(item.get('cancelled_count') or 0)}")
            if int(item.get('dropped_count') or 0):
                counts.append(f"drop{int(item.get('dropped_count') or 0)}")
            spectrum = ', '.join(str(val) for val in (item.get('status_spectrum') or []) if val)
            details = ' '.join(counts) if counts else 'events logged'
            suffix = f" | spectrum {spectrum}" if spectrum else ''
            lines.append(f"- {item.get('label') or item.get('module')}: {details} | last {item.get('last_status') or 'unknown'}{suffix}")
    recent_activity = scheduler.get('recent_activity') or []
    cancelled = [item for item in recent_activity if str(item.get('status') or '').strip().lower() == 'cancelled']
    if cancelled:
        last = cancelled[0]
        lines.append(f"Last cancel: {last.get('label') or last.get('task_key')} ({str(last.get('reason') or '').replace('_', ' ')})")
    return '\n'.join(lines)


def _resource_pressure_level(stats):
    system_mem = float(stats.get('system_mem') or 0.0)
    if system_mem >= 92.0:
        return 'hard'
    if system_mem >= 82.0:
        return 'soft'
    return 'normal'


def _resource_delta_label(delta_bytes):
    if abs(delta_bytes) < (128 * 1024 * 1024):
        return 'stable'
    return 'rising' if delta_bytes > 0 else 'falling'


def _percent_delta_label(delta_value):
    if abs(delta_value) < 1.0:
        return 'stable'
    return 'rising' if delta_value > 0 else 'falling'


def _history_window(history, window_size):
    if not history:
        return []
    if len(history) <= window_size:
        return list(history)
    return list(history[-window_size:])


def _compute_resource_trends(history):
    if not history:
        return {
            'samples': 0,
            'short': {'direction': 'unknown', 'ram_delta_bytes': 0, 'system_ram_delta_percent': 0.0, 'seconds': 0},
            'long': {'direction': 'unknown', 'ram_delta_bytes': 0, 'system_ram_delta_percent': 0.0, 'seconds': 0},
            'summary': 'No resource trend data yet.',
        }

    def _window_trend(items):
        if len(items) < 2:
            return {'direction': 'stable', 'ram_delta_bytes': 0, 'system_ram_delta_percent': 0.0, 'seconds': 0}
        first = items[0]
        last = items[-1]
        ram_delta = int(last.get('ina_ram_bytes') or 0) - int(first.get('ina_ram_bytes') or 0)
        sys_delta = float(last.get('system_ram_percent') or 0.0) - float(first.get('system_ram_percent') or 0.0)
        direction = _resource_delta_label(ram_delta)
        try:
            start_ts = datetime.fromisoformat(str(first.get('timestamp')))
            end_ts = datetime.fromisoformat(str(last.get('timestamp')))
            seconds = max(0, int((end_ts - start_ts).total_seconds()))
        except Exception:
            seconds = 0
        return {
            'direction': direction,
            'ram_delta_bytes': ram_delta,
            'system_ram_delta_percent': round(sys_delta, 1),
            'seconds': seconds,
        }

    short = _window_trend(_history_window(history, RESOURCE_TREND_SHORT_SAMPLES))
    long = _window_trend(_history_window(history, RESOURCE_TREND_LONG_SAMPLES))
    latest = history[-1]
    largest = (latest.get('top_modules') or [{}])[0]
    largest_name = largest.get('name') or 'unknown module'
    summary = (
        f"RAM trend is {short['direction']} over the short window ({_format_ram_value(abs(short['ram_delta_bytes']))} over {short['seconds']}s) "
        f"and {long['direction']} over the long window ({_format_ram_value(abs(long['ram_delta_bytes']))} over {long['seconds']}s). "
        f"System RAM is {_percent_delta_label(long['system_ram_delta_percent'])}. Biggest current holder: {largest_name}."
    )
    return {
        'samples': len(history),
        'short': short,
        'long': long,
        'summary': summary,
    }


def _summarize_resource_usage(stats):
    level = _resource_pressure_level(stats)
    top_modules = (stats.get('modules') or [])[:3]
    top_text = ', '.join(
        f"{item['name']} {_format_ram_value(item['mem_bytes'])}"
        for item in top_modules
    ) if top_modules else 'no child modules detected'
    cpu_modules = [item for item in (stats.get('cpu_modules') or [])[:3] if float(item.get('cpu') or 0.0) > 0.0]
    cpu_text = ', '.join(
        f"{item['name']} {float(item.get('cpu') or 0.0):.1f}%"
        for item in cpu_modules
    ) if cpu_modules else 'waiting for the next CPU sample'
    if level == 'hard':
        note = 'Pressure is high. Large RAM holders should shed memory before new work starts.'
    elif level == 'soft':
        note = 'Pressure is rising. Watch the largest RAM holders first.'
    else:
        note = 'Pressure is stable. The largest RAM holders are the clearest optimisation targets.'
    return (
        f"Total Ina RAM is {_format_ram_value(int(stats.get('mem_bytes') or 0))} across {int(stats.get('processes') or 0)} process(es). "
        f"System RAM is at {float(stats.get('system_mem') or 0.0):.1f}%. "
        f"Top RAM: {top_text}. Top CPU: {cpu_text}. {note}"
    )


def _publish_resource_snapshot(stats):
    global _last_resource_publish_ts, _last_resource_publish_key
    now = time.time()
    top_modules = []
    for item in (stats.get('modules') or [])[:6]:
        top_modules.append({
            'name': item['name'],
            'ram_bytes': int(item['mem_bytes']),
            'ram_human': _format_ram_value(item['mem_bytes']),
            'cpu_percent': round(float(item['cpu']), 1),
            'threads': int(item['threads']),
            'processes': int(item['processes']),
            'pids': [int(pid) for pid in (item.get('pids') or [])[:8]],
        })
    top_cpu_modules = []
    for item in (stats.get('cpu_modules') or [])[:6]:
        top_cpu_modules.append({
            'name': item['name'],
            'ram_bytes': int(item['mem_bytes']),
            'ram_human': _format_ram_value(item['mem_bytes']),
            'cpu_percent': round(float(item['cpu']), 1),
            'threads': int(item['threads']),
            'processes': int(item['processes']),
            'pids': [int(pid) for pid in (item.get('pids') or [])[:8]],
        })
    top_cpu_processes = []
    for item in (stats.get('process_rows') or [])[:6]:
        top_cpu_processes.append({
            'pid': int(item['pid']),
            'name': item['name'],
            'ram_bytes': int(item['mem_bytes']),
            'ram_human': _format_ram_value(item['mem_bytes']),
            'cpu_percent': round(float(item['cpu']), 1),
            'threads': int(item['threads']),
        })
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = _summarize_resource_usage(stats)
    scheduler = _scheduler_snapshot()
    if top_cpu_modules and float(top_cpu_modules[0].get('cpu_percent') or 0.0) > 0.0:
        top_cpu = top_cpu_modules[0]
        optimization_hint = (
            f"Start CPU optimisation with {top_cpu['name']} because it is currently the hottest process group "
            f"at {float(top_cpu.get('cpu_percent') or 0.0):.1f}% CPU."
        )
    elif top_modules:
        optimization_hint = f"Start optimisation with {top_modules[0]['name']} because it currently holds the most RAM."
    else:
        optimization_hint = 'No active child modules detected; measure the largest live process before optimising.'
    key = (
        round(float(stats.get('cpu') or 0.0) / 5.0) * 5,
        round(float(stats.get('mem_bytes') or 0) / (1024.0 ** 3), 2),
        round(float(stats.get('system_mem') or 0.0), 1),
        tuple((item['name'], item['ram_human']) for item in top_modules[:3]),
        tuple((item['name'], round(float(item['cpu_percent'] or 0.0) / 5.0) * 5) for item in top_cpu_modules[:3]),
        int(scheduler.get('queue_depth') or 0),
        int(scheduler.get('running_count') or 0),
        tuple(item.get('task_key') for item in (scheduler.get('running') or [])[:3]),
        tuple(item.get('task_key') for item in (scheduler.get('next_slots') or [])[:3]),
    )
    if _last_resource_publish_key == key and _last_resource_publish_ts and (now - _last_resource_publish_ts) < RESOURCE_PUBLISH_INTERVAL_SEC:
        return
    history = get_inastate('resource_vitals_history') or []
    if not isinstance(history, list):
        history = []
    history.append({
        'timestamp': timestamp,
        'pressure_level': _resource_pressure_level(stats),
        'ina_ram_bytes': int(stats.get('mem_bytes') or 0),
        'system_ram_percent': round(float(stats.get('system_mem') or 0.0), 1),
        'top_modules': top_modules[:3],
        'top_cpu_modules': top_cpu_modules[:3],
    })
    history = history[-RESOURCE_HISTORY_MAX_SAMPLES:]
    trends = _compute_resource_trends(history)
    update_inastate('resource_vitals_history', history)
    update_inastate('resource_vitals', {
        'timestamp': timestamp,
        'source': 'gui_vitals',
        'pressure_level': _resource_pressure_level(stats),
        'ina_cpu_percent': round(float(stats.get('cpu') or 0.0), 1),
        'ina_ram_bytes': int(stats.get('mem_bytes') or 0),
        'ina_ram_human': _format_ram_value(int(stats.get('mem_bytes') or 0)),
        'system_cpu_percent': round(float(stats.get('system_cpu') or 0.0), 1),
        'system_ram_percent': round(float(stats.get('system_mem') or 0.0), 1),
        'process_count': int(stats.get('processes') or 0),
        'thread_count': int(stats.get('threads') or 0),
        'top_modules': top_modules,
        'top_cpu_modules': top_cpu_modules,
        'top_cpu_processes': top_cpu_processes,
        'summary': summary,
        'optimization_hint': optimization_hint,
        'trend': trends,
        'process_scheduler': {
            'available': bool(scheduler.get('available')),
            'summary': scheduler.get('summary'),
            'learning_hint': scheduler.get('learning_hint'),
            'running': scheduler.get('running'),
            'next_slots': scheduler.get('next_slots'),
            'last_decisions': scheduler.get('last_decisions'),
            'recent_activity': scheduler.get('recent_activity'),
            'module_history': scheduler.get('module_history'),
            'history_window_hours': round(float(scheduler.get('history_window_hours') or 24.0), 2),
            'cancelled_count': int(scheduler.get('cancelled_count') or 0),
            'queue_depth': int(scheduler.get('queue_depth') or 0),
            'running_count': int(scheduler.get('running_count') or 0),
            'blocked_count': int(scheduler.get('blocked_count') or 0),
            'memory_guard_level': scheduler.get('memory_guard_level'),
            'cpu_percent': round(float(scheduler.get('cpu_percent') or 0.0), 1),
            'gpu_utilization_percent': round(float(scheduler.get('gpu_utilization_percent') or 0.0), 1),
            'gpu_available': bool(scheduler.get('gpu_available')),
        },
    })
    _last_resource_publish_ts = now
    _last_resource_publish_key = key


def _collect_usage_snapshot():
    stats = {
        'cpu': 0.0,
        'mem_bytes': 0,
        'threads': 0,
        'processes': 0,
        'system_cpu': 0.0,
        'system_mem': 0.0,
        'modules': [],
        'cpu_modules': [],
        'process_rows': [],
    }
    modules = {}
    process_rows = []
    live_keys = set()
    now = time.monotonic()

    for proc in _ina_processes():
        try:
            key = _process_cpu_sample_key(proc)
            live_keys.add(key)
            cpu = _sample_process_cpu_percent(proc, now=now)
            mem_bytes = int(proc.memory_info().rss)
            threads = proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        stats['cpu'] += cpu
        stats['mem_bytes'] += mem_bytes
        stats['threads'] += threads
        stats['processes'] += 1
        name = _module_process_name(proc)
        pid = int(proc.pid)
        process_rows.append({'pid': pid, 'name': name, 'cpu': cpu, 'mem_bytes': mem_bytes, 'threads': threads})
        bucket = modules.setdefault(name, {'name': name, 'cpu': 0.0, 'mem_bytes': 0, 'threads': 0, 'processes': 0, 'pids': []})
        bucket['cpu'] += cpu
        bucket['mem_bytes'] += mem_bytes
        bucket['threads'] += threads
        bucket['processes'] += 1
        bucket['pids'].append(pid)

    try:
        stats['system_cpu'] = psutil.cpu_percent(interval=None)
    except Exception:
        stats['system_cpu'] = 0.0

    try:
        stats['system_mem'] = psutil.virtual_memory().percent
    except Exception:
        stats['system_mem'] = 0.0

    for item in modules.values():
        item['pids'] = sorted(dict.fromkeys(item.get('pids') or []))
    stats['modules'] = sorted(
        modules.values(),
        key=lambda item: (item['mem_bytes'], item['cpu'], item['name']),
        reverse=True,
    )
    stats['cpu_modules'] = sorted(
        modules.values(),
        key=lambda item: (item['cpu'], item['mem_bytes'], item['name']),
        reverse=True,
    )
    stats['process_rows'] = sorted(
        process_rows,
        key=lambda item: (item['cpu'], item['mem_bytes'], item['name'], item['pid']),
        reverse=True,
    )
    _prune_process_cpu_samples(live_keys)
    return stats


def _refresh_energy_label():
    if energy_status_var is None:
        return
    current = _clamp_value(get_inastate("current_energy") or 0.0, 0.0, 1.0)
    energy_status_var.set(f"Current energy: {current:.3f}")


def _refresh_nutrition_section():
    global hunger_status_var, fitness_status_var, nutrition_info_var, last_meal_status_var, metabolic_status_var
    if hunger_status_var is None:
        return
    hunger = _clamp_value(get_inastate("hunger_level") or 0.6, 0.0, 1.0)
    fitness = _clamp_value(get_inastate("fitness_level") or 0.55, 0.0, 1.0)
    hunger_status_var.set(f"Hunger: {hunger:.3f}")
    fitness_status_var.set(f"Fitness: {fitness:.3f}")
    status = get_inastate("nutrition_status") or {}
    eff = status.get("metabolic_efficiency")
    if eff is None:
        metabolic_status_var.set("Metabolic efficiency: --")
    else:
        metabolic_status_var.set(f"Metabolic efficiency: {float(eff):.3f}")
    last_meal = status.get("last_meal")
    if last_meal:
        label = last_meal.get("label") or last_meal.get("name", "--")
        reason = last_meal.get("reason", "?")
        timestamp = last_meal.get("timestamp", "--")
        last_meal_status_var.set(f"Last meal: {label} ({reason}) @ {timestamp}")
    else:
        last_meal_status_var.set("Last meal: --")
    pending_offers = status.get("pending_offers") or []
    if offer_status_var is not None:
        if pending_offers:
            lines = []
            for offer in pending_offers[:3]:
                label = offer.get("label") or offer.get("name", "--")
                note = offer.get("note")
                stamp = offer.get("offered_at", "--")
                line = f"{label} @ {stamp}"
                if note:
                    line += f" — {note}"
                lines.append(line)
            if len(pending_offers) > 3:
                lines.append(f"...and {len(pending_offers) - 3} more")
            offer_status_var.set("Offers:\n" + "\n".join(lines))
        else:
            offer_status_var.set("Offers: none pending")
    options = status.get("options") or []
    if options:
        summary_lines = []
        for opt in options[:4]:
            ready = "✓" if opt.get("cooldown_ready") else "…"
            summary_lines.append(
                f"{opt.get('label', opt.get('name'))}: {opt.get('score', 0.0):.2f} {ready}"
            )
        nutrition_info_var.set("\n".join(summary_lines))
    else:
        nutrition_info_var.set("Meal scores pending…")


def _request_meal_from_gui(meal_name: str):
    if not request_meal(meal_name, reason="gui"):
        messagebox.showerror("Nutrition", f"Unable to schedule {meal_name} right now.")
        return
    append_status(f"[Vitals] Requested {meal_name.replace('_', ' ')} for Ina.\n")
    _refresh_nutrition_section()


def _offer_meal_from_gui(meal_name: str):
    note = offer_note_var.get().strip() if offer_note_var else ""
    note_value = note or None
    if not offer_meal(meal_name, note=note_value):
        messagebox.showerror("Nutrition", f"Unable to log offer {meal_name}.")
        return
    append_status(f"[Vitals] Offered {meal_name.replace('_', ' ')} to Ina.\n")
    _refresh_nutrition_section()

def _apply_energy_value(value=None, reason="manual"):
    if energy_var is None:
        return
    val = _clamp_value(value if value is not None else energy_var.get(), 0.0, 1.0)
    energy_var.set(round(val, 3))
    update_inastate("current_energy", round(val, 3))
    _refresh_energy_label()
    append_status(f"[Vitals] Energy set to {val:.3f} ({reason}).\n")

def _nudge_energy(delta):
    if energy_var is None:
        return
    current = _clamp_value(energy_var.get(), 0.0, 1.0)
    _apply_energy_value(current + delta, reason="nudge")

def _current_emotion_seed():
    snapshot = get_inastate("emotion_snapshot") or {}
    values = snapshot.get("values") if isinstance(snapshot, dict) else None
    if not isinstance(values, dict):
        values = snapshot if isinstance(snapshot, dict) else {}
    if not values:
        try:
            cfg = load_config()
            child = cfg.get("current_child", "Inazuma_Yagami")
            values = load_baseline(child)
        except Exception:
            values = {}
    cleaned = {}
    for key in EMOTION_SLIDERS:
        cleaned[key] = _clamp_value(values.get(key, 0.0), -1.0, 1.0)
    return cleaned

def _reload_emotion_sliders():
    if not emotion_vars:
        return
    seed = _current_emotion_seed()
    for key, var in emotion_vars.items():
        var.set(seed.get(key, 0.0))

def _apply_emotion_sliders():
    if not emotion_vars:
        return
    values = {name: _clamp_value(var.get(), -1.0, 1.0) for name, var in emotion_vars.items()}
    mode = get_inastate("mode", "awake") or "awake"
    processed = process_emotion(values, mode=mode)
    ts = datetime.now(timezone.utc).isoformat()

    update_inastate("emotion_snapshot", {"timestamp": ts, "mode": mode, "values": processed})
    update_inastate("last_emotion_update", ts)
    append_status(f"[Vitals] Emotion sliders applied (mode={mode}).\n")

def _update_usage_labels():
    if vitals_window is None or not vitals_window.winfo_exists():
        return

    stats = _collect_usage_snapshot()
    summary = _summarize_resource_usage(stats)

    if _usage_labels.get('ina_cpu'):
        _usage_labels['ina_cpu'].config(text=f"Ina CPU (sum): {stats['cpu']:.1f}%")
    if _usage_labels.get('ina_mem'):
        _usage_labels['ina_mem'].config(text=f"Ina RAM: {_format_ram_value(stats['mem_bytes'])}")
    if _usage_labels.get('ina_threads'):
        _usage_labels['ina_threads'].config(text=f"Threads: {stats['threads']}  ·  Processes: {stats['processes']}")
    if _usage_labels.get('sys_cpu'):
        _usage_labels['sys_cpu'].config(text=f"System CPU: {stats['system_cpu']:.1f}%")
    if _usage_labels.get('sys_mem'):
        _usage_labels['sys_mem'].config(text=f"System RAM: {stats['system_mem']:.1f}%")
    history = get_inastate('resource_vitals_history') or []
    trends = _compute_resource_trends(history if isinstance(history, list) else [])
    scheduler = _scheduler_snapshot()
    if _usage_labels.get('resource_note'):
        _usage_labels['resource_note'].config(text=summary)
    if _usage_labels.get('resource_trend'):
        _usage_labels['resource_trend'].config(text=trends.get('summary', 'No resource trend data yet.'))
    if _usage_labels.get('module_mem'):
        _usage_labels['module_mem'].config(text=_format_process_usage(stats.get('process_rows', [])))
    if _usage_labels.get('scheduler_note'):
        _usage_labels['scheduler_note'].config(text=_summarize_scheduler_state(scheduler))
    if _usage_labels.get('scheduler_slots'):
        _usage_labels['scheduler_slots'].config(text=_format_scheduler_slots(scheduler))

    _publish_resource_snapshot(stats)
    _refresh_energy_label()
    _refresh_nutrition_section()
    vitals_window.after(1500, _update_usage_labels)

def open_vitals_window():
    global vitals_window, _usage_labels, energy_var, energy_status_var, emotion_vars
    global hunger_status_var, fitness_status_var, nutrition_info_var, last_meal_status_var, metabolic_status_var
    global offer_status_var, offer_note_var

    if vitals_window is not None and vitals_window.winfo_exists():
        vitals_window.lift()
        return

    vitals_window = tk.Toplevel(root)
    vitals_window.title('Ina Vitals & Control')
    vitals_window.geometry('760x760')
    vitals_window.minsize(640, 520)

    outer = tk.Frame(vitals_window)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    content = tk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=content, anchor='nw')

    def _sync_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox('all'))

    def _sync_canvas_width(event):
        canvas.itemconfigure(canvas_window, width=event.width)

    def _scroll_vitals_canvas(event):
        if getattr(event, 'delta', 0):
            step = -1 if event.delta > 0 else 1
        elif getattr(event, 'num', None) == 4:
            step = -1
        elif getattr(event, 'num', None) == 5:
            step = 1
        else:
            return None
        canvas.yview_scroll(step, 'units')
        return 'break'

    def _bind_vitals_mousewheel(widget):
        widget.bind('<MouseWheel>', _scroll_vitals_canvas, add='+')
        widget.bind('<Button-4>', _scroll_vitals_canvas, add='+')
        widget.bind('<Button-5>', _scroll_vitals_canvas, add='+')
        for child in widget.winfo_children():
            _bind_vitals_mousewheel(child)

    content.bind('<Configure>', _sync_scroll_region)
    canvas.bind('<Configure>', _sync_canvas_width)

    usage_frame = tk.LabelFrame(content, text='Resource usage (Ina process tree)')
    usage_frame.pack(fill=tk.X, padx=10, pady=(10, 6))

    _usage_labels = {
        'ina_cpu': tk.Label(usage_frame, text='Ina CPU (sum): --'),
        'ina_mem': tk.Label(usage_frame, text='Ina RAM: --'),
        'ina_threads': tk.Label(usage_frame, text='Threads: --'),
        'sys_cpu': tk.Label(usage_frame, text='System CPU: --'),
        'sys_mem': tk.Label(usage_frame, text='System RAM: --'),
        'resource_note': tk.Label(
            usage_frame,
            text='Readable summary: gathering…',
            justify=tk.LEFT,
            anchor='w',
            wraplength=680,
        ),
        'resource_trend': tk.Label(
            usage_frame,
            text='Trend summary: gathering…',
            justify=tk.LEFT,
            anchor='w',
            wraplength=680,
        ),
        'module_mem': tk.Label(
            usage_frame,
            text='Per-process modules: gathering…',
            justify=tk.LEFT,
            anchor='w',
            font='TkFixedFont',
        ),
        'scheduler_note': tk.Label(
            usage_frame,
            text='Scheduler summary: gathering…',
            justify=tk.LEFT,
            anchor='w',
            wraplength=680,
        ),
        'scheduler_slots': tk.Label(
            usage_frame,
            text='Scheduler queue: gathering…',
            justify=tk.LEFT,
            anchor='w',
            font='TkFixedFont',
        ),
    }

    _usage_labels['ina_cpu'].pack(anchor='w')
    _usage_labels['ina_mem'].pack(anchor='w')
    _usage_labels['ina_threads'].pack(anchor='w')
    _usage_labels['sys_cpu'].pack(anchor='w', pady=(4, 0))
    _usage_labels['sys_mem'].pack(anchor='w')
    tk.Label(usage_frame, text='Readable summary for Ina').pack(anchor='w', pady=(8, 0))
    _usage_labels['resource_note'].pack(anchor='w', fill=tk.X, pady=(2, 0))
    tk.Label(usage_frame, text='Trend summary for Ina').pack(anchor='w', pady=(8, 0))
    _usage_labels['resource_trend'].pack(anchor='w', fill=tk.X, pady=(2, 0))
    tk.Label(usage_frame, text='Per-process RAM / CPU (PID shown)').pack(anchor='w', pady=(8, 0))
    _usage_labels['module_mem'].pack(anchor='w', fill=tk.X, pady=(2, 0))
    tk.Label(usage_frame, text='Process scheduler for Ina').pack(anchor='w', pady=(8, 0))
    _usage_labels['scheduler_note'].pack(anchor='w', fill=tk.X, pady=(2, 0))
    _usage_labels['scheduler_slots'].pack(anchor='w', fill=tk.X, pady=(2, 0))

    energy_frame = tk.LabelFrame(content, text='Energy')
    energy_frame.pack(fill=tk.X, padx=10, pady=(6, 6))

    energy_var = tk.DoubleVar(value=_clamp_value(get_inastate('current_energy') or 0.5, 0.0, 1.0))
    energy_status_var = tk.StringVar(value='Current energy: --')

    energy_row = tk.Frame(energy_frame)
    energy_row.pack(fill=tk.X, padx=5, pady=4)
    tk.Label(energy_row, text='Energy (0-1)').pack(side=tk.LEFT)
    tk.Scale(
        energy_row,
        from_=0.0,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        variable=energy_var,
        length=320,
    ).pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
    tk.Label(energy_row, textvariable=energy_status_var).pack(side=tk.LEFT, padx=5)

    energy_buttons = tk.Frame(energy_frame)
    energy_buttons.pack(fill=tk.X, padx=5, pady=(0, 4))
    tk.Button(energy_buttons, text='Nudge -0.05', command=lambda: _nudge_energy(-0.05)).pack(side=tk.LEFT, padx=4)
    tk.Button(energy_buttons, text='Apply', command=lambda: _apply_energy_value(reason='slider')).pack(side=tk.LEFT, padx=4)
    tk.Button(energy_buttons, text='Nudge +0.05', command=lambda: _nudge_energy(0.05)).pack(side=tk.LEFT, padx=4)
    tk.Button(
        energy_buttons,
        text='Reload',
        command=lambda: energy_var.set(_clamp_value(get_inastate('current_energy') or 0.5, 0.0, 1.0)),
    ).pack(side=tk.LEFT, padx=4)

    nutrition_frame = tk.LabelFrame(content, text='Nutrition & Fitness')
    nutrition_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

    hunger_status_var = tk.StringVar(value='Hunger: --')
    fitness_status_var = tk.StringVar(value='Fitness: --')
    metabolic_status_var = tk.StringVar(value='Metabolic efficiency: --')
    last_meal_status_var = tk.StringVar(value='Last meal: --')
    nutrition_info_var = tk.StringVar(value='Meal scores pending…')
    offer_status_var = tk.StringVar(value='Offers: --')
    offer_note_var = tk.StringVar()

    tk.Label(nutrition_frame, textvariable=hunger_status_var).pack(anchor='w', padx=6, pady=(4, 0))
    tk.Label(nutrition_frame, textvariable=fitness_status_var).pack(anchor='w', padx=6)
    tk.Label(nutrition_frame, textvariable=metabolic_status_var).pack(anchor='w', padx=6)
    tk.Label(nutrition_frame, textvariable=last_meal_status_var, wraplength=520, justify=tk.LEFT).pack(anchor='w', padx=6, pady=(0, 4))

    tk.Label(nutrition_frame, text='Meal gate scores:').pack(anchor='w', padx=6)
    tk.Label(nutrition_frame, textvariable=nutrition_info_var, justify=tk.LEFT, wraplength=520).pack(anchor='w', padx=6, pady=(0, 4))

    tk.Label(nutrition_frame, textvariable=offer_status_var, justify=tk.LEFT, wraplength=520).pack(anchor='w', padx=6, pady=(0, 4))

    note_row = tk.Frame(nutrition_frame)
    note_row.pack(fill=tk.X, padx=6, pady=(0, 4))
    tk.Label(note_row, text='Offer note:').pack(side=tk.LEFT)
    tk.Entry(note_row, textvariable=offer_note_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    meal_buttons = tk.Frame(nutrition_frame)
    meal_buttons.pack(fill=tk.X, padx=6, pady=(0, 4))
    tk.Button(meal_buttons, text='Snack', command=lambda: _request_meal_from_gui('snack')).pack(side=tk.LEFT, padx=4)
    tk.Button(meal_buttons, text='Small Meal', command=lambda: _request_meal_from_gui('small_meal')).pack(side=tk.LEFT, padx=4)
    tk.Button(meal_buttons, text='Meal', command=lambda: _request_meal_from_gui('meal')).pack(side=tk.LEFT, padx=4)
    tk.Button(meal_buttons, text='Large Meal', command=lambda: _request_meal_from_gui('large_meal')).pack(side=tk.LEFT, padx=4)
    tk.Button(meal_buttons, text='Reload', command=_refresh_nutrition_section).pack(side=tk.RIGHT, padx=4)

    offer_buttons = tk.Frame(nutrition_frame)
    offer_buttons.pack(fill=tk.X, padx=6, pady=(0, 4))
    tk.Button(offer_buttons, text='Offer Snack', command=lambda: _offer_meal_from_gui('snack')).pack(side=tk.LEFT, padx=4)
    tk.Button(offer_buttons, text='Offer Small Meal', command=lambda: _offer_meal_from_gui('small_meal')).pack(side=tk.LEFT, padx=4)
    tk.Button(offer_buttons, text='Offer Meal', command=lambda: _offer_meal_from_gui('meal')).pack(side=tk.LEFT, padx=4)
    tk.Button(offer_buttons, text='Offer Large Meal', command=lambda: _offer_meal_from_gui('large_meal')).pack(side=tk.LEFT, padx=4)

    emotion_frame = tk.LabelFrame(content, text='Emotion sliders (-1 to 1)')
    emotion_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

    emotion_vars = {}
    seed = _current_emotion_seed()
    for idx, slider_name in enumerate(EMOTION_SLIDERS):
        col = idx // 12
        row = idx % 12
        tk.Label(emotion_frame, text=slider_name).grid(row=row, column=col * 2, sticky='w', padx=4, pady=2)
        var = tk.DoubleVar(value=seed.get(slider_name, 0.0))
        emotion_vars[slider_name] = var
        tk.Scale(
            emotion_frame,
            from_=-1.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=var,
            length=240,
        ).grid(row=row, column=col * 2 + 1, sticky='ew', padx=4, pady=2)

    for col in range(4):
        emotion_frame.columnconfigure(col, weight=1)

    controls_row = tk.Frame(emotion_frame)
    controls_row.grid(row=12, column=0, columnspan=4, sticky='ew', padx=4, pady=(8, 2))
    tk.Button(controls_row, text='Reload from state', command=_reload_emotion_sliders).pack(side=tk.LEFT, padx=4)
    tk.Button(controls_row, text='Apply to Ina', command=_apply_emotion_sliders).pack(side=tk.LEFT, padx=4)

    _bind_vitals_mousewheel(vitals_window)

    _prime_usage_counters()
    _refresh_energy_label()
    _refresh_nutrition_section()
    vitals_window.after(500, _update_usage_labels)

def open_logs():
    log_path = Path("AI_Children") / load_config().get("current_child", "Inazuma_Yagami") / "memory" / "self_questions.json"
    if not log_path.exists():
        messagebox.showinfo("Log", "No log found.")
        return
    with open(log_path, "r") as f:
        log = json.load(f)
    display = tk.Toplevel()
    display.title("Self Questions")
    text = tk.Text(display, height=20, width=80)
    text.pack()
    for entry in log[-20:]:
        question = entry.get("question", "unknown")
        first = entry.get("first_asked") or entry.get("timestamp") or ""
        count = entry.get("count", 1)
        resolved = entry.get("resolved_at")
        line = f"{first} — {question} (×{count})"
        if resolved:
            line += f" [resolved {resolved}]"
        text.insert(tk.END, line + "\n")
    text.config(state=tk.DISABLED)

def emergency_shutdown():
    global model_running
    model_running = False
    now = datetime.now(timezone.utc).isoformat()
    shutdown_payload = {
        "timestamp": now,
        "source": "gui",
        "mode": "emergency",
        "clean": False,
    }
    update_inastate("shutdown_intent", shutdown_payload)
    update_inastate("last_shutdown", shutdown_payload)
    update_inastate("dreaming", False)
    update_inastate("runtime_disruption", True)

    print("[Emergency] Triggering immediate shutdown...")

    os.system("pkill -f model_manager.py")
    os.system("pkill -f dreamstate.py")
    os.system("pkill -f boredom_state.py")
    os.system("pkill -f meditation_state.py")
    os.system("pkill -f early_comm.py")
    os.system("pkill -f audio_listener.py")
    os.system("pkill -f vision_window.py")
    os.system("pkill -f birth_system.py")
    os.system("pkill -f emotion_engine.py")
    os.system("pkill -f emotion_map.py")
    os.system("pkill -f expression_log.py")
    os.system("pkill -f fractal_multidimensional_transformers.py")
    os.system("pkill -f fragmentation_engine.py")
    os.system("pkill -f inject_birth_fragment.py")
    os.system("pkill -f instinct_engine.py")
    os.system("pkill -f logic_engine.py")
    os.system("pkill -f meaning_map.py")
    os.system("pkill -f memory_graph.py")
    os.system("pkill -f precision_evolution.py")
    os.system("pkill -f predictive_layer.py")
    os.system("pkill -f pretrain_logic.py")
    os.system("pkill -f raw_file_manager.py")
    os.system("pkill -f train_fragments.py")
    os.system("pkill -f who_am_i.py")

    print("[Emergency] Core modules halted.")


def tuck_in():
    try:
        safe_popen(["python", "dreamstate.py"], label="Dream", verbose=True)
    except Exception as e:
        messagebox.showerror("Dream Error", f"Failed to launch dreamstate: {e}")


def wake_up():
    refresh_config()
    child = config.get("current_child", "default_child")

    build_fractal_memory(child)
    status_box.insert(tk.END, "[Wake] Running post-wake self-reflection...\n")
    stream_subprocess_to_status([sys.executable, "who_am_i.py"], label="Self-Reflection")

    time.sleep(1)
    status_box.insert(tk.END, "[Wake] Resuming communication loop...\n")
    safe_popen([sys.executable, "early_comm.py"], verbose=False)



def reboot_model():
    refresh_config()

    if config.get("dreaming", False):
        status_box.insert(tk.END, "[Reboot] Ina is dreaming — tucking her in properly first...\n")
        status_box.see(tk.END)
        tuck_in()
        time.sleep(2)

    status_box.insert(tk.END, "[Reboot] Initiating reboot sequence...\n")
    status_box.see(tk.END)

    emergency_shutdown()
    time.sleep(1)

    start_model()
    time.sleep(3)

    wake_up()
    status_box.insert(tk.END, "[Reboot] Reboot complete.\n")
    status_box.see(tk.END)



def quit_program():
    if model_running:
        status_box.insert(tk.END, "Quit blocked: model is currently running.\n")
        status_box.see(tk.END)
        messagebox.showwarning("Model Active", "A model is currently running. Please stop it before quitting.")
        return

    if messagebox.askokcancel("Quit Program", "Are you sure you want to quit?"):
        status_box.insert(tk.END, "Quit Program confirmed. Exiting...\n")
        status_box.see(tk.END)
        save_config()
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        root.quit()
    else:
        status_box.insert(tk.END, "Quit Program cancelled.\n")
        status_box.see(tk.END)

root = tk.Tk()
root.title("Ina")

refresh_config()
if 'geometry' in config:
    root.geometry(config['geometry'])
else:
    root.geometry("500x450")
root.minsize(500, 450)

book_path_var = tk.StringVar(value=config.get("book_folder_path", ""))
music_path_var = tk.StringVar(value=config.get("music_folder_path", ""))

menu_bar = Menu(root)
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Birth New Model", command=birth_new_model)
file_menu.add_command(label="Load Child", command=load_child)
menu_bar.add_cascade(label="File", menu=file_menu)

options_menu = Menu(menu_bar, tearoff=0)
options_menu.add_command(label="Save/Load Config", command=save_load_config)
options_menu.add_command(label="Exceptions List", command=exceptions_list)
options_menu.add_command(label="Precision Settings", command=precision_settings)
options_menu.add_command(label="Timers", command=open_timers_config)
options_menu.add_command(label="Audio Devices", command=open_audio_devices_window)
options_menu.add_command(label="Vitals / Emotions", command=open_vitals_window)
options_menu.add_command(label="Signal High Memory", command=lambda: signal_memory_too_high(source="gui_menu"))
menu_bar.add_cascade(label="Options", menu=options_menu)

root.config(menu=menu_bar)

main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

tk.Label(main_frame, text="AIs Online:", font=("Helvetica", 14)).pack(pady=(5, 0))
canvas = tk.Canvas(main_frame, width=100, height=100, bg="black", highlightthickness=0)
canvas.pack(pady=10)
canvas.create_oval(10, 10, 90, 90, outline="green", width=2)
ai_text_id = canvas.create_text(50, 50, text="0", fill="green", font=("Helvetica", 16, "bold"))

tk.Label(main_frame, text=f"Target: {config.get('current_child', 'None')}", font=("Helvetica", 10, "italic")).pack()

status_frame = tk.Frame(main_frame)
status_frame.pack(expand=True, fill=tk.BOTH, pady=5)
scrollbar = tk.Scrollbar(status_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

status_box = tk.Text(status_frame, height=6, wrap=tk.WORD, yscrollcommand=scrollbar.set)
status_box.pack(expand=True, fill=tk.BOTH)
status_box.tag_config("error", foreground="red")
scrollbar.config(command=status_box.yview)

paths_container = tk.Frame(main_frame)
paths_container.pack(fill=tk.X, pady=(5, 10))

book_row = tk.Frame(paths_container)
book_row.pack(fill=tk.X, pady=2)
tk.Label(book_row, text="Book Folder:").pack(side=tk.LEFT)
book_entry = tk.Entry(book_row, textvariable=book_path_var)
book_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
book_entry.bind("<FocusOut>", commit_book_folder)
book_entry.bind("<Return>", commit_book_folder)
tk.Button(book_row, text="Browse...", command=browse_book_folder).pack(side=tk.LEFT, padx=(5, 0))

music_row = tk.Frame(paths_container)
music_row.pack(fill=tk.X, pady=2)
tk.Label(music_row, text="Music Folder:").pack(side=tk.LEFT)
music_entry = tk.Entry(music_row, textvariable=music_path_var)
music_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
music_entry.bind("<FocusOut>", commit_music_folder)
music_entry.bind("<Return>", commit_music_folder)
tk.Button(music_row, text="Browse...", command=browse_music_folder).pack(side=tk.LEFT, padx=(5, 0))

button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=10)
tk.Button(button_frame, text="Start Model", command=start_model, width=15).grid(row=0, column=3, padx=5)
tk.Button(button_frame, text="Emergency Shutdown", command=emergency_shutdown, width=15).grid(row=0, column=4, padx=5)
tk.Button(button_frame, text="Quit Program", command=quit_program, width=15).grid(row=0, column=5, padx=5)

if config.get("is_root", False):
    tk.Button(button_frame, text="Pretrain", command=pretrain_mode, width=15).grid(row=0, column=6, padx=5)
    tk.Button(button_frame, text="EEG", command=open_eeg_view, width=15).grid(row=0, column=7, padx=5)

button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=10)

tk.Button(button_frame, text="Reboot", command=reboot_model, width=15).grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Tuck In", command=tuck_in, width=15).grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Wake Up", command=wake_up, width=15).grid(row=0, column=2, padx=5)
tk.Button(button_frame, text="Clear Log", command=clear_status_log, width=15).grid(row=0, column=3, padx=5)
tk.Button(button_frame, text="Vitals + Sliders", command=open_vitals_window, width=17).grid(row=0, column=4, padx=5)
tk.Button(button_frame, text="View Self Questions", command=open_logs, width=20).grid(row=0, column=5, padx=5)




root.bind_all("<Control-Shift-M>", _shortcut_memory_too_high)


root.protocol("WM_DELETE_WINDOW", quit_program)
status_log_server()
root.mainloop()
