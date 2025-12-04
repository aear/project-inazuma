import tkinter as tk
from tkinter import messagebox
import json
import os

CONFIG_FILE = "config.json"

DEFAULT_TIMERS = {
    "expression_interval": 30,
    "reflection_interval": 300,
    "dream_cycle_interval": 120
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

class TimersWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Timer Settings")
        self.geometry("300x250")
        self.config_data = load_config()
        self.timers = self.config_data.get("timers", DEFAULT_TIMERS.copy())
        self.entries = {}

        tk.Label(self, text="Adjust Timers (seconds)", font=("Helvetica", 12)).pack(pady=10)

        for idx, (key, value) in enumerate(self.timers.items()):
            frame = tk.Frame(self)
            frame.pack(pady=5)
            tk.Label(frame, text=key.replace("_", " ").title() + ": ").pack(side=tk.LEFT)
            entry = tk.Entry(frame)
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT)
            self.entries[key] = entry

        save_btn = tk.Button(self, text="Save", command=self.save)
        save_btn.pack(pady=15)

    def save(self):
        for key, entry in self.entries.items():
            try:
                self.timers[key] = int(entry.get())
            except ValueError:
                messagebox.showerror("Invalid Input", f"Value for {key} must be an integer.")
                return

        self.config_data["timers"] = self.timers
        save_config(self.config_data)
        messagebox.showinfo("Success", "Timers updated.")
        self.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    TimersWindow(root)
    root.mainloop()