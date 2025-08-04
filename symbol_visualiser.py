
# === symbol_visualiser.py ===
# Converts Ina's generated symbols into images for visual training

import os
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from model_manager import load_config

def load_generated_symbols(child):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data.get("self_generated_symbols", [])
        except:
            return []

def render_symbol_to_image(symbol_text, font_size=128, image_size=(256, 256)):
    image = Image.new("L", image_size, color=255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except:
        font = ImageFont.load_default()
    w, h = draw.textsize(symbol_text, font=font)
    x = (image_size[0] - w) // 2
    y = (image_size[1] - h) // 2
    draw.text((x, y), symbol_text, font=font, fill=0)
    return image

def save_symbol_images(symbols, child):
    base = Path("AI_Children") / child / "memory" / "vision_session" / "generated_symbols"
    base.mkdir(parents=True, exist_ok=True)
    manifest = []

    for entry in symbols:
        symbol = entry.get("symbol", "")
        summary = entry.get("meaning", "")
        if not symbol:
            continue

        image = render_symbol_to_image(symbol)
        filename = f"sym_{symbol.encode().hex()}.png"
        image.save(base / filename)

        manifest.append({
            "symbol": symbol,
            "summary": summary,
            "image": filename,
            "timestamp": entry.get("timestamp")
        })

    with open(base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"[SymbolVisualiser] Saved {len(manifest)} symbol images.")

def main():
    config = load_config()
    child = config.get("current_child", "default_child")
    symbols = load_generated_symbols(child)
    save_symbol_images(symbols, child)

if __name__ == "__main__":
    main()
