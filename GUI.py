import tkinter as tk
from tkinter import Menu, messagebox, filedialog
import json
import os
import sys
import subprocess
import psutil
import shutil
from pathlib import Path
from model_manager import update_inastate
import threading
import time
from memory_graph import build_fractal_memory
import platform
from birth_system import boot


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
                                status_box.insert(tk.END, msg)
                                status_box.see(tk.END)
                else:
                    with open(STATUS_PIPE_PATH, "r") as pipe:
                        for msg in pipe:
                            if msg.strip():
                                status_box.insert(tk.END, msg)
                                status_box.see(tk.END)
            except Exception as e:
                status_box.insert(tk.END, f"[Pipe Error] {e}\n")
                status_box.see(tk.END)
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
        status_box.insert(tk.END, f"[{label}] Starting...\n")
        status_box.see(tk.END)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(process.stdout.readline, ''):
                if line:
                    status_box.insert(tk.END, f"[{label}] {line}")
                    status_box.see(tk.END)
            process.stdout.close()
            process.wait()
            status_box.insert(tk.END, f"[{label}] Completed.\n")
            status_box.see(tk.END)
        except Exception as e:
            status_box.insert(tk.END, f"[{label}] ERROR: {e}\n")
            status_box.see(tk.END)

    threading.Thread(target=stream_output, daemon=True).start()



CONFIG_FILE = "config.json"
model_running = False


def safe_popen(cmd):
    try:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        status_box.insert(tk.END, f"[ERROR] Failed to start {' '.join(map(str, cmd))}: {e}\n")
        status_box.see(tk.END)

def refresh_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass


def save_config():
    config_path = CONFIG_FILE
    current = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                current = json.load(f)
        except json.JSONDecodeError:
            pass
    current["geometry"] = root.winfo_geometry()
    with open(config_path, "w") as f:
        json.dump(current, f, indent=4)

def birth_new_model():
    status_box.insert(tk.END, "Opening Birth Certificate window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "birth_certificate.py"])

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
    safe_popen([sys.executable, "exception_window.py"])

def precision_settings():
    status_box.insert(tk.END, "Opening Precision Settings window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "precision_window.py"])

def open_timers_config():
    status_box.insert(tk.END, "Opening Timers configuration.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "timers_window.py"])


def pretrain_mode():
    status_box.insert(tk.END, "Entering Pretrain mode...\n")
    status_box.see(tk.END)

    def stream_pretrain():
        # Fetch child from the current configuration
        config = load_config()
        child = config.get("current_child", "Inazuma_Yagami")

        status_box.insert(tk.END, f"[Pretrain] Using child: {child}\n")
        status_box.see(tk.END)

        # Now pass child to pretrain_logic.py as an argument
        try:
            process = subprocess.Popen(
                [sys.executable, "pretrain_logic.py", child],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in iter(process.stdout.readline, ''):
                if line:
                    status_box.insert(tk.END, f"[Pretrain] {line}")
                    status_box.see(tk.END)
            process.stdout.close()
            process.wait()
        except Exception as e:
            status_box.insert(tk.END, f"[Pretrain] ERROR: {e}\n")
            status_box.see(tk.END)
            return

        status_box.insert(tk.END, "[Pretrain] Finished pretraining.\n")
        status_box.see(tk.END)

    threading.Thread(target=stream_pretrain, daemon=True).start()



def open_eeg_view():
    status_box.insert(tk.END, "Opening EEG window.\n")
    status_box.see(tk.END)
    safe_popen([sys.executable, "EEG.py"])

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
        safe_popen(["python", "dreamstate.py"])
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
    safe_popen([sys.executable, "early_comm.py"])



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
scrollbar.config(command=status_box.yview)

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
root.mainloop()
# After root.mainloop setup
status_log_server()
