# precision_window.py

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import platform
import subprocess
import shutil
import logging

CONFIG_FILE = "config.json"

def load_main_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def save_main_config(update):
    config_path = "config.json"
    current = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                current = json.load(f)
        except json.JSONDecodeError:
            pass
    current.update(update)
    with open(config_path, "w") as f:
        json.dump(current, f, indent=4)


# Setup logging
logging.basicConfig(
    filename="precision_window.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

# Try to import torch for CUDA detection.
try:
    import torch
except ImportError:
    torch = None
    logging.warning("Torch library not found; NVIDIA GPU detection disabled.")

CONFIG_FILE = "precision_config.json"

def load_precision_config():
    """Load saved precision configuration from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                logging.info("Loaded precision configuration.")
                return config
        except json.JSONDecodeError as e:
            logging.error("JSON decode error in precision config: %s", e)
    else:
        logging.info("Precision configuration file not found; using defaults.")
    return {}

def save_precision_config(config):
    """Save precision configuration to a JSON file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        logging.info("Saved precision configuration: %s", config)
    except Exception as e:
        logging.error("Error saving precision configuration: %s", e)

def get_cpu_name():
    """
    Return a descriptive CPU name.
    First attempt using platform.uname(). If that yields no usable string,
    try reading '/proc/cpuinfo' on Linux.
    """
    uname = platform.uname()
    cpu = uname.processor
    if cpu and cpu.strip():
        logging.info("CPU name detected from platform.uname(): %s", cpu)
        return cpu
    # Fallback for Linux systems: try reading /proc/cpuinfo.
    if os.name == "posix" and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        if cpu_model:
                            logging.info("CPU name detected from /proc/cpuinfo: %s", cpu_model)
                            return cpu_model
        except Exception as e:
            logging.error("Error reading /proc/cpuinfo: %s", e)
    # If still no name, use system and machine info.
    cpu = f"{uname.system} {uname.machine}"
    logging.info("Falling back to system/machine info for CPU name: %s", cpu)
    return cpu or "CPU"

def get_amd_gpu_info():
    """
    Try to detect AMD GPUs using 'rocm-smi' if available.
    Returns a dictionary where keys are device names and values are precision info.
    """
    amd_gpus = {}
    if shutil.which("rocm-smi") is None:
        logging.info("'rocm-smi' not found; AMD GPU detection skipped.")
        return amd_gpus

    try:
        # Run rocm-smi to show product names.
        output = subprocess.check_output(["rocm-smi", "--showproductname"], universal_newlines=True)
        logging.info("rocm-smi output:\n%s", output)
        for line in output.splitlines():
            if "GPU" in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    identifier = parts[0].strip()  # e.g., "GPU[0]"
                    try:
                        gpu_index = int(identifier.strip("GPU[]"))
                    except Exception:
                        gpu_index = None
                    gpu_name = parts[-1].strip() if parts[-1].strip() else "AMD GPU"
                    key = f"GPU {gpu_index} (AMD {gpu_name})" if gpu_index is not None else f"AMD GPU ({gpu_name})"
                    int_list = ["8-bit", "16-bit", "32-bit"]
                    float_list = ["FP16", "FP32"]
                    if "XTX" in gpu_name.upper():
                        float_list.append("FP64")
                    bf_list = []
                    amd_gpus[key] = {
                        "Integer": int_list,
                        "Floating Point": float_list,
                        "Brain Floats": bf_list
                    }
                    logging.info("Detected AMD GPU: %s with precisions: %s", key, amd_gpus[key])
        return amd_gpus
    except Exception as e:
        logging.error("Error detecting AMD GPUs: %s", e)
        return {}

def get_hardware_info():
    """
    Detect hardware precision capabilities for CPU, NVIDIA (via PyTorch), and AMD GPUs.
    Returns a dictionary where keys are device names and values are dicts of supported precisions.
    """
    hardware = {}

    # CPU information.
    cpu_name = get_cpu_name()
    hardware[f"{cpu_name} (CPU)"] = {
        "Integer": ["8-bit", "16-bit", "32-bit", "64-bit"],
        "Floating Point": ["FP16", "FP32", "FP64"],
        "Brain Floats": []  # Typically not available on CPUs.
    }
    logging.info("Added CPU info: %s", hardware[f"{cpu_name} (CPU)"])

    # GPU information using torch (assumes CUDA, normally NVIDIA).
    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_name = props.name
            int_list = ["8-bit", "16-bit", "32-bit"]
            float_list = ["FP16", "FP32"]
            if props.major >= 3:
                float_list.append("FP64")
            bf_list = []
            if props.major >= 8:
                bf_list.append("BF16")
            key = f"GPU {i} (NVIDIA {gpu_name})"
            hardware[key] = {
                "Integer": int_list,
                "Floating Point": float_list,
                "Brain Floats": bf_list
            }
            logging.info("Added NVIDIA GPU %d: %s", i, hardware[key])
    else:
        logging.info("No NVIDIA GPUs detected or torch unavailable.")

    # AMD GPU detection.
    amd_gpus = get_amd_gpu_info()
    hardware.update(amd_gpus)
    if amd_gpus:
        logging.info("AMD GPUs detected: %s", amd_gpus)
    else:
        logging.info("No AMD GPUs detected.")

    return hardware

def compute_hardware_max_precision(hardware_info):
    """
    Compute the maximum floating point precision available among the detected hardware.
    Returns an integer (e.g. 64 if any device supports FP64, else 32 or 16).
    """
    max_prec = 0
    for device, info in hardware_info.items():
        for fp in info.get("Floating Point", []):
            if "64" in fp:
                max_prec = max(max_prec, 64)
            elif "32" in fp:
                max_prec = max(max_prec, 32)
            elif "16" in fp:
                max_prec = max(max_prec, 16)
    if max_prec == 0:
        max_prec = 32
    logging.info("Computed hardware max precision: %d bits", max_prec)
    return max_prec

class PrecisionWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Precision Settings")
        self.geometry("650x500")
        self.minsize(650, 500)
        self.resizable(True, True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load geometry
        self.config_data = load_main_config()
        self.apply_geometry()

        # Query hardware precision capabilities.
        self.hardware_info = get_hardware_info()
        self.hardware_max_precision = compute_hardware_max_precision(self.hardware_info)

        # Load saved precision settings (or default to hardware max).
        config = load_precision_config()
        self.software_simulation_enabled = config.get("software_simulation", False)
        self.max_precision_setting = config.get("max_precision", self.hardware_max_precision)

        self.create_widgets()


    def create_widgets(self):
        # Frame for hardware-supported precision details.
        details_frame = ttk.LabelFrame(self, text="Hardware Supported Precisions")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Treeview for displaying supported precisions.
        columns = ("device", "integer", "floating", "brain_floats")
        self.tree = ttk.Treeview(details_frame, columns=columns, show="headings", height=6)
        self.tree.heading("device", text="Device")
        self.tree.heading("integer", text="Integer Precisions")
        self.tree.heading("floating", text="Floating Point Precisions")
        self.tree.heading("brain_floats", text="Brain Floats")
        self.tree.column("device", width=150, anchor=tk.CENTER)
        self.tree.column("integer", width=180, anchor=tk.CENTER)
        self.tree.column("floating", width=180, anchor=tk.CENTER)
        self.tree.column("brain_floats", width=150, anchor=tk.CENTER)

        for device, precisions in self.hardware_info.items():
            int_prec = ", ".join(precisions.get("Integer", []))
            float_prec = ", ".join(precisions.get("Floating Point", []))
            bf_prec = ", ".join(precisions.get("Brain Floats", []))
            self.tree.insert("", tk.END, values=(device, int_prec, float_prec, bf_prec))
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame for software simulated precision settings.
        software_frame = ttk.LabelFrame(self, text="Software Simulated Precision")
        software_frame.pack(fill=tk.X, padx=10, pady=10)

        # Checkbox for enabling software simulation.
        self.software_var = tk.BooleanVar(value=self.software_simulation_enabled)
        self.software_checkbox = ttk.Checkbutton(
            software_frame,
            text="Enable Software Simulated Precision",
            variable=self.software_var
        )
        self.software_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # Spinbox for maximum precision input.
        max_label = ttk.Label(software_frame, text="Max Precision (bits):")
        max_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_precision_var = tk.IntVar(value=self.max_precision_setting)
        self.max_precision_spinbox = tk.Spinbox(
            software_frame, from_=16, to=128,
            textvariable=self.max_precision_var,
            width=8, command=self.check_precision
        )
        self.max_precision_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        self.max_precision_var.trace_add("write", lambda *args: self.check_precision())

        # Label for informational messages.
        self.info_label = ttk.Label(software_frame, text="")
        self.info_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # Button frame for saving settings and closing.
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        save_button = ttk.Button(button_frame, text="Save", command=self.save_settings)
        save_button.pack(side=tk.LEFT, padx=5)
        close_button = ttk.Button(button_frame, text="Close", command=self.destroy)
        close_button.pack(side=tk.LEFT, padx=5)

        self.check_precision()

    def check_precision(self):
        """
        Disable software simulation if the user's max precision is less than or equal
        to the hardware's max supported precision.
        """
        try:
            max_prec = self.max_precision_var.get()
        except tk.TclError:
            return
        if max_prec <= self.hardware_max_precision:
            self.software_checkbox.config(state=tk.DISABLED)
            self.software_var.set(False)
            self.info_label.config(
                text=f"Hardware precision of {self.hardware_max_precision} bits is sufficient."
            )
            logging.info("Software simulation disabled; max precision (%d) <= hardware max (%d)", max_prec, self.hardware_max_precision)
        else:
            self.software_checkbox.config(state=tk.NORMAL)
            self.info_label.config(
                text=f"Software simulation available for precision above {self.hardware_max_precision} bits."
            )
            logging.info("Software simulation enabled option; max precision (%d) > hardware max (%d)", max_prec, self.hardware_max_precision)

    def save_settings(self):
        """
        Save the precision settings to a JSON file and close the window.
        """
        config = {
            "software_simulation": self.software_var.get(),
            "max_precision": self.max_precision_var.get()
        }
        save_precision_config(config)
        messagebox.showinfo(
            "Settings Saved",
            f"Precision settings saved:\nSoftware Simulation: {'Enabled' if config['software_simulation'] else 'Disabled'}\nMax Precision: {config['max_precision']} bits"
        )
        logging.info("Precision settings saved and window closed.")
        self.destroy()

    def apply_geometry(self):
        geom = self.config_data.get("precision_window_geometry")
        if geom:
            self.geometry(geom)

    def on_close(self):
        save_main_config({"precision_window_geometry": self.geometry()})
        self.destroy()


if __name__ == "__main__":
    app = PrecisionWindow()
    app.mainloop()
