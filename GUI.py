import tkinter as tk
from tkinter import Menu, messagebox, filedialog
import json
import os
import sys
from safe_popen import safe_popen
import psutil
import shutil
from pathlib import Path
from model_manager import update_inastate
import threading
import time
from memory_graph import build_fractal_memory
import platform
from birth_system import boot

def append_status(msg, tag=None):
    """Safely append to the status box from any thread."""
    def _append():
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



CONFIG_FILE = "config.json"
CONFIG_DEFAULTS = {
    "book_folder_path": "",
    "music_folder_path": "",
}
config = dict(CONFIG_DEFAULTS)
book_path_var = None
music_path_var = None
model_running = False


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
    status_box.insert(tk.END, "Start Button clicked.\n")
    status_box.insert(tk.END, "Launching Birth System...\n")
    status_box.see(tk.END)
    child = config.get("current_child", "default_child")

    boot(child)

    update_ai_count_label()


def load_config():
    path = Path("config.json")
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

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
        text.insert(tk.END, f"{entry['timestamp']} — {entry['question']}")
    text.config(state=tk.DISABLED)

def emergency_shutdown():
    global model_running
    model_running = False
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
tk.Button(button_frame, text="View Self Questions", command=open_logs, width=20)




root.protocol("WM_DELETE_WINDOW", quit_program)
status_log_server()
root.mainloop()
