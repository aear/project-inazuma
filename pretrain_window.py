# pretrain_window.py

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
import threading
import pretrain_logic


CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def save_config(update):
    config_path = CONFIG_FILE
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

class PretrainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pretraining Interface")
        self.geometry("600x500")
        self.minsize(600, 500)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.config_data = load_config()
        self.apply_geometry()
        self.birth_path = self.config_data.get("birth_certificate_path", "")


        self.selected_files = []

        self.create_widgets()

    def create_widgets(self):
        # Top Frame for File Selection
        file_frame = tk.LabelFrame(self, text="Training Files")
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        self.file_listbox = tk.Listbox(file_frame, height=5)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(file_frame, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        button_frame = tk.Frame(self)
        button_frame.pack(pady=(0, 10))

        add_button = tk.Button(button_frame, text="Add Files", command=self.add_files)
        add_button.pack(side=tk.LEFT, padx=5)

        clear_button = tk.Button(button_frame, text="Clear List", command=self.clear_list)
        clear_button.pack(side=tk.LEFT, padx=5)

        # Birth Certificate Selection
        birth_frame = tk.LabelFrame(self, text="Birth Certificate")
        birth_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.birth_var = tk.StringVar(value=self.birth_path)
        birth_entry = ttk.Entry(birth_frame, textvariable=self.birth_var)
        birth_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        browse_btn = tk.Button(birth_frame, text="Browse", command=self.browse_birth_certificate)
        browse_btn.pack(side=tk.LEFT, padx=5)


        # Status Box
        self.status_box = tk.Text(self, height=10, wrap=tk.WORD)
        self.status_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        # Start/Save Buttons
        action_frame = tk.Frame(self)
        action_frame.pack(pady=5)

        start_btn = tk.Button(action_frame, text="Start", command=self.start_pretrain)
        start_btn.pack(side=tk.LEFT, padx=5)

        load_btn = tk.Button(action_frame, text="Load List", command=self.load_file_list)
        load_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(action_frame, text="Save List", command=self.save_file_list)
        save_btn.pack(side=tk.LEFT, padx=5)

        quit_btn = tk.Button(action_frame, text="Close", command=self.on_close)
        quit_btn.pack(side=tk.LEFT, padx=5)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select Training Files")
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_listbox.insert(tk.END, file)
                self.status_box.insert(tk.END, f"Added: {file}\n")

    def clear_list(self):
        self.selected_files = []
        self.file_listbox.delete(0, tk.END)
        self.status_box.insert(tk.END, "Cleared file list.\n")

    def browse_birth_certificate(self):
        path = filedialog.askopenfilename(title="Select Birth Certificate", filetypes=[("JSON files", "*.json")])
        if path:
            self.birth_var.set(path)
            self.birth_path = path
            self.status_box.insert(tk.END, f"Selected birth certificate: {path}\n")


    def save_file_list(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if save_path:
            with open(save_path, "w") as f:
                json.dump(self.selected_files, f, indent=4)
            self.status_box.insert(tk.END, f"Saved file list to: {save_path}\n")

    def load_file_list(self):
        load_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if load_path:
            try:
                with open(load_path, "r") as f:
                    loaded_files = json.load(f)
                if isinstance(loaded_files, list):
                    self.selected_files = loaded_files
                    self.file_listbox.delete(0, tk.END)
                    for file in loaded_files:
                        self.file_listbox.insert(tk.END, file)
                    self.status_box.insert(tk.END, f"Loaded file list from: {load_path}\n")
                else:
                    self.status_box.insert(tk.END, f"[ERROR] Loaded file was not a list.\n")
            except Exception as e:
                self.status_box.insert(tk.END, f"[ERROR] Failed to load file list: {e}\n")


    def start_pretrain(self):
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select training files first.")
            return

        self.status_box.insert(tk.END, f"Pretraining started on {len(self.selected_files)} files...\n")
        self.status_box.see(tk.END)

        thread = threading.Thread(target=self.run_pipeline)
        thread.start()

    def run_pipeline(self):
        try:
            self.update_status("Stage 1: Running Pretraining Logic...\n")
            pretrain_logic.run_all(birth_path=self.birth_path)
            self.update_status("Pretraining complete.\n")
        except Exception as e:
            self.update_status(f"[ERROR] Pretraining failed: {str(e)}\n")

    def update_status(self, text):
        self.status_box.insert(tk.END, text)
        self.status_box.see(tk.END)


    def apply_geometry(self):
        geom = self.config_data.get("pretrain_window_geometry")
        if geom:
            self.geometry(geom)

    def on_close(self):
        save_config({"pretrain_window_geometry": self.geometry()})
        self.destroy()

if __name__ == "__main__":
    app = PretrainWindow()
    app.mainloop()
