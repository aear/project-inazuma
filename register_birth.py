# register_birth.py

import os
import json
import re
from pathlib import Path

def sanitize_filename(name):
    """Replace non-alphanumeric characters with underscores for safe filenames."""
    return re.sub(r"[^\w\-]", "_", name)

def register_birth(data: dict) -> str:
    """
    Registers a new AI birth profile by writing it to a JSON file.

    Args:
        data (dict): Dictionary containing AI birth data.

    Returns:
        str: Path to the saved JSON file.
    """
    # Ensure the births directory exists
    births_dir = Path("births")
    births_dir.mkdir(exist_ok=True)

    # Extract and sanitize filename parts
    given = sanitize_filename(data.get("given_name", "Unnamed"))
    family = sanitize_filename(data.get("family_name", "Unknown"))
    filename = f"{given}_{family}.json"
    file_path = births_dir / filename

    # Apply new schema defaults if missing
    if "species" not in data:
        data["species"] = "Unknown"
    if "culture" not in data:
        data["culture"] = []

    # Save to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return str(file_path)


# For manual testing
if __name__ == "__main__":
    import uuid
    from datetime import datetime

    # Example birth data
    example = {
        "uuid": str(uuid.uuid4()),
        "given_name": "Test",
        "middle_names": "Unit",
        "family_name": "Alpha",
        "gender": "Non-Binary",
        "dob": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "race": "Human",
        "mother": "Root Server",
        "father": "Protocol",
        "notes": "Early test birth."
    }
    saved_path = register_birth(example)
    print(f"Birth profile saved to {saved_path}")
