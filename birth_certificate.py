# birth_certificate.py

import tkinter as tk
from tkinter import ttk, messagebox
import uuid
import json
import os
from datetime import datetime
import register_birth

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


class BirthCertificateWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Birth Certificate")
        self.geometry("500x600")
        self.minsize(500, 600)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.config_data = load_config()
        self.apply_geometry()

        self.current_dob = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cultures = []
        self.create_widgets()

    def apply_geometry(self):
        geom = self.config_data.get("birth_certificate_geometry")
        if geom:
            self.geometry(geom)

    def on_close(self):
        self.config_data["birth_certificate_geometry"] = self.geometry()
        save_config({"birth_certificate_geometry": self.geometry()})
        self.destroy()

    def add_culture(self):
        new_culture = self.culture_entry.get().strip()
        if new_culture and new_culture not in self.cultures:
            self.cultures.append(new_culture)
            self.culture_listbox.insert(tk.END, new_culture)
            self.culture_entry.delete(0, tk.END)

    def remove_culture(self):
        selected = self.culture_listbox.curselection()
        if selected:
            idx = selected[0]
            self.cultures.pop(idx)
            self.culture_listbox.delete(idx)

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 5}
        self.given_name_entry = ttk.Entry(self)
        self.middle_names_entry = ttk.Entry(self)
        self.family_name_entry = ttk.Entry(self)
        self.gender_var = tk.StringVar(value="Unknown")
        self.mother_entry = ttk.Entry(self)
        self.father_entry = ttk.Entry(self)
        self.notes_text = tk.Text(self, height=6, wrap=tk.WORD)

        ttk.Label(self, text="Given Name:").pack(**padding)
        self.given_name_entry.pack(fill=tk.X, **padding)
        ttk.Label(self, text="Middle Name(s):").pack(**padding)
        self.middle_names_entry.pack(fill=tk.X, **padding)
        ttk.Label(self, text="Family Name:").pack(**padding)
        self.family_name_entry.pack(fill=tk.X, **padding)
        ttk.Label(self, text="Gender:").pack(**padding)
        ttk.Combobox(self, textvariable=self.gender_var, values=["Female", "Male", "Non-Binary", "Unknown"]).pack(fill=tk.X, **padding)
        ttk.Label(self, text="Species:").pack(**padding)
        self.species_var = tk.StringVar(value="Human")
        ttk.Combobox(self, textvariable=self.species_var, values=["Human", "Uedan", "Other"]).pack(fill=tk.X, **padding)

        ttk.Label(self, text="Culture(s):").pack(**padding)
        culture_frame = tk.Frame(self)
        culture_frame.pack(fill=tk.X, **padding)
        self.culture_entry = ttk.Entry(culture_frame)
        self.culture_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(culture_frame, text="+", command=self.add_culture).pack(side=tk.LEFT, padx=5)

        self.culture_listbox = tk.Listbox(self, height=3)
        self.culture_listbox.pack(fill=tk.X, padx=10)

        ttk.Button(self, text="Remove Selected Culture", command=self.remove_culture).pack(pady=(0, 10))

        ttk.Label(self, text="Mother's Name:").pack(**padding)
        self.mother_entry.pack(fill=tk.X, **padding)
        ttk.Label(self, text="Father's Name:").pack(**padding)
        self.father_entry.pack(fill=tk.X, **padding)
        ttk.Label(self, text="Notes:").pack(**padding)
        self.notes_text.pack(fill=tk.BOTH, expand=True, **padding)
        ttk.Button(self, text="Submit", command=self.submit).pack(pady=10)

    def submit(self):
        given_name = self.given_name_entry.get().strip()
        middle_names = self.middle_names_entry.get().strip()
        family_name = self.family_name_entry.get().strip()
        gender = self.gender_var.get()
        dob = self.current_dob
        species = self.species_var.get()
        cultures = list(self.culture_listbox.get(0, tk.END))
        mother = self.mother_entry.get().strip()
        father = self.father_entry.get().strip()
        notes = self.notes_text.get("1.0", tk.END).strip()

        if not given_name or not family_name:
            messagebox.showwarning("Input Error", "Please enter both Given Name and Family Name.")
            return

        birth_uuid = str(uuid.uuid4())
        birth_data = {
            "uuid": birth_uuid,
            "given_name": given_name,
            "middle_names": middle_names,
            "family_name": family_name,
            "gender": gender,
            "dob": dob,
            "species": species,
            "culture": cultures,
            "mother": mother,
            "father": father,
            "notes": notes
        }

        full_name = f"{given_name}_{family_name}".strip()
        ai_dir = Path("AI_Children") / full_name
        memory_dir = ai_dir / "memory"

        try:
            if not ai_dir.exists():
                ai_dir.mkdir(parents=True)
                memory_dir.mkdir(parents=True)
                (memory_dir / "fragments").mkdir(parents=True)

            # Save birth certificate
            cert_path = ai_dir / "birth_certificate.json"
            with open(cert_path, "w", encoding="utf-8") as f:
                json.dump(birth_data, f, indent=4)

            # Update current config
            save_config({"current_child": full_name})

            cert_info = (
                f"Full Name: {given_name} {' ' + middle_names if middle_names else ''} {family_name}".strip() + "\n"
                f"Gender: {gender}\nDate/Time of Birth: {dob}\n"
                f"Species: {species}\nCulture: {', '.join(cultures)}\nUUID: {birth_uuid}\n"
                f"Parents: Mother: {mother or 'N/A'}, Father: {father or 'N/A'}\n"
                f"Notes: {notes}\nSaved to: {cert_path}"
            )
            print("Birth Certificate Created:\n" + cert_info)
            messagebox.showinfo("Birth Registered", f"Birth profile saved:\n\n{cert_path}")
            self.on_close()
        except Exception as e:
            messagebox.showerror("Registration Error", f"Failed to register birth: {str(e)}")





if __name__ == "__main__":
    app = BirthCertificateWindow()
    app.mainloop()
