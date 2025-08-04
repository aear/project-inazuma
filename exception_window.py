# exception_window.py

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import json
import os

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



# File to store the exceptions
EXCEPTIONS_FILE = "exceptions.json"

def load_exceptions():
    """Load exceptions from a JSON file. Returns a list."""
    if os.path.exists(EXCEPTIONS_FILE):
        try:
            with open(EXCEPTIONS_FILE, "r") as f:
                data = json.load(f)
                # Ensure we return a list
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass
    return []

def save_exceptions(exceptions):
    """Save the exceptions list to a JSON file."""
    with open(EXCEPTIONS_FILE, "w") as f:
        json.dump(exceptions, f, indent=4)

class ExceptionsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exceptions Manager")
        self.geometry("600x400")
        self.minsize(600, 400)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.config_data = load_config()
        self.apply_geometry()


        # Current list of exceptions
        self.exceptions = load_exceptions()

        # Create UI elements
        self.create_widgets()
        # Populate the listbox with loaded exceptions
        self.populate_listbox()

    def create_widgets(self):
        # Frame for the entry controls
        control_frame = tk.Frame(self)
        control_frame.pack(pady=10, padx=10, fill=tk.X)

        # Label and OptionMenu for exception type
        tk.Label(control_frame, text="Exception Type:").grid(row=0, column=0, sticky="w")
        self.exception_types = ["Program", "Folder", "Web Address"]
        self.type_var = tk.StringVar(value=self.exception_types[0])
        type_menu = ttk.OptionMenu(control_frame, self.type_var, self.exception_types[0], *self.exception_types, command=self.update_browse_state)
        type_menu.grid(row=0, column=1, padx=5, sticky="w")

        # Label and Entry for exception target
        tk.Label(control_frame, text="Path/URL:").grid(row=1, column=0, sticky="w")
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(control_frame, textvariable=self.entry_var, width=40)
        self.entry.grid(row=1, column=1, padx=5, sticky="w")

        # Browse button (only enabled for Program and Folder)
        self.browse_button = tk.Button(control_frame, text="Browse...", command=self.browse_exception)
        self.browse_button.grid(row=1, column=2, padx=5)
        self.update_browse_state(self.exception_types[0])

        # Button to add an exception
        add_button = tk.Button(control_frame, text="Add Exception", command=self.add_exception)
        add_button.grid(row=2, column=0, columnspan=3, pady=5)

        # Info label about wildcard support
        wildcard_info = tk.Label(control_frame, text="* Wildcards are supported, e.g., *.gelbooru.com/*", font=("Helvetica", 8), fg="gray")
        wildcard_info.grid(row=3, column=0, columnspan=3, sticky="w", padx=5)

        # Listbox to display exceptions with a scrollbar
        listbox_frame = tk.Frame(self)
        listbox_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Frame for action buttons at the bottom
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        remove_button = tk.Button(button_frame, text="Remove Selected", command=self.remove_selected)
        remove_button.grid(row=0, column=0, padx=5)

        save_button = tk.Button(button_frame, text="Save", command=self.save_and_close)
        save_button.grid(row=0, column=1, padx=5)

        cancel_button = tk.Button(button_frame, text="Cancel", command=self.destroy)
        cancel_button.grid(row=0, column=2, padx=5)

    def update_browse_state(self, selected_type):
        """Enable or disable the Browse button based on the exception type."""
        if selected_type in ["Program", "Folder"]:
            self.browse_button.config(state=tk.NORMAL)
        else:
            self.browse_button.config(state=tk.DISABLED)

    def browse_exception(self):
        """Open a file dialog for Program or a directory dialog for Folder."""
        ex_type = self.type_var.get()
        if ex_type == "Program":
            # For programs, let the user select a file
            path = filedialog.askopenfilename(title="Select Program")
            if path:
                self.entry_var.set(path)
        elif ex_type == "Folder":
            # For folders, let the user select a directory
            folder = filedialog.askdirectory(title="Select Folder")
            if folder:
                self.entry_var.set(folder)
        else:
            messagebox.showinfo("Browse Not Available", "Browsing is not available for Web Addresses.")

    def populate_listbox(self):
        """Load exceptions into the listbox."""
        self.listbox.delete(0, tk.END)
        for item in self.exceptions:
            # Each item is a dictionary with 'type' and 'value'
            display = f"{item['type']}: {item['value']}"
            self.listbox.insert(tk.END, display)

    def add_exception(self):
        """Add a new exception from the entry fields."""
        exception_type = self.type_var.get()
        exception_value = self.entry_var.get().strip()
        if not exception_value:
            messagebox.showwarning("Input Error", "Please enter a valid path, directory, or URL.")
            return

        new_exception = {"type": exception_type, "value": exception_value}
        self.exceptions.append(new_exception)
        self.populate_listbox()
        self.entry_var.set("")  # Clear the entry field

    def remove_selected(self):
        """Remove the currently selected exception from the list."""
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select an exception to remove.")
            return
        index = selection[0]
        del self.exceptions[index]
        self.populate_listbox()

    def save_and_close(self):
        """Save the exceptions list to the file and close the window."""
        save_exceptions(self.exceptions)
        messagebox.showinfo("Saved", "Exceptions saved successfully.")
        self.destroy()

    def apply_geometry(self):
        geom = self.config_data.get("exception_window_geometry")
        if geom:
            self.geometry(geom)

    def on_close(self):
        save_config({"exception_window_geometry": self.geometry()})
        self.destroy()


if __name__ == "__main__":
    app = ExceptionsWindow()
    app.mainloop()
