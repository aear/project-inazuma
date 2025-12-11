
import os
import json
import random
from datetime import datetime, timezone
from pathlib import Path

# Procedural symbol parts
EMOTION_GLYPHS = {
    "calm": "∘", "tension": "∇", "curiosity": "μ", "trust": "λ", "fear": "ψ", "anger": "Ω"
}
MODULATION_GLYPHS = {
    "soft": "·", "moderate": "⇌", "sharp": "∴", "pulse": "∆", "spiral": "⊙"
}
CONCEPT_GLYPHS = {
    "self": "ν", "pattern": "Ξ", "truth": "φ", "change": "∵", "unknown": "∅"
}

# Extra marks used to vary length/texture of generated symbols
ACCENT_GLYPHS = ["·", ":", "~", "ː", "⁂", "↺", "↯"]
# Allow occasional ASCII letters/digits so Ina can lean on them if she wants.
ALPHANUMERIC_GLYPHS = list("abcdefghijklmnopqrstuvwxyz0123456789")

# Procedural map
def generate_symbol_from_parts(emotion_key, mod_key, concept_key, length=None):
    """
    Build a symbol from emotion/modulation/concept glyphs.
    Length can vary (defaults to 2–5 chars) so symbols are not locked to 3 chars.
    """
    base_parts = [
        EMOTION_GLYPHS[emotion_key],
        MODULATION_GLYPHS[mod_key],
        CONCEPT_GLYPHS[concept_key],
    ]

    target_len = length if length is not None else random.choice([2, 3, 3, 4, 4, 5])
    target_len = max(2, min(target_len, 6))  # keep symbols compact

    # Always keep emotion + concept, optionally keep modulation if length allows
    symbol_parts = [base_parts[0], base_parts[2]]
    if target_len >= 3:
        symbol_parts.insert(1, base_parts[1])
    elif random.random() < 0.5:
        symbol_parts[1] = base_parts[1]  # occasionally swap concept for modulation when length=2

    # Add accent glyphs or repeat base glyphs until we reach the target length
    while len(symbol_parts) < target_len:
        extra_pool = ACCENT_GLYPHS + base_parts
        # Small chance to mix in letters/numbers; not forced, just available.
        if random.random() < 0.35:
            extra_pool = extra_pool + ALPHANUMERIC_GLYPHS
        symbol_parts.append(random.choice(extra_pool))

    return "".join(symbol_parts)

def procedural_combinations():
    combos = []
    for e in EMOTION_GLYPHS:
        for m in MODULATION_GLYPHS:
            for c in CONCEPT_GLYPHS:
                symbol = generate_symbol_from_parts(e, m, c)
                meaning = f"{m} {e} about {c}"
                combos.append({
                    "symbol": symbol,
                    "meaning": meaning,
                    "components": {"emotion": e, "modulation": m, "concept": c},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "origin": "self_generated_procedural"
                })
    return combos

def enrich_symbols(symbols, transformer):
    enriched = []
    for entry in symbols:
        emotions = {
            "novelty": 0.6 if entry["components"]["concept"] == "unknown" else 0.3,
            "care": 0.4 if entry["components"]["emotion"] == "trust" else 0.2,
            "intensity": 0.6 if entry["components"]["modulation"] == "sharp" else 0.3
        }
        result = transformer.encode({"emotions": emotions})
        clarity = round(sum(result["vector"]) / len(result["vector"]), 4)

        entry["emotions"] = emotions
        entry["clarity"] = clarity
        entry["tags"] = ["symbolic", "procedural", "self_generated"]
        enriched.append(entry)
    return enriched

def load_self_reflection(child):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_self_reflection(child, data):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_as_fragments(child, enriched):
    frag_dir = Path("AI_Children") / child / "memory" / "fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    for entry in enriched:
        frag = {
            "id": f"frag_selfsymbol_{entry['symbol']}",
            "summary": entry["meaning"],
            "tags": entry["tags"],
            "timestamp": entry["timestamp"],
            "source": "symbol_generator",
            "emotions": entry["emotions"],
            "clarity": entry["clarity"]
        }
        path = frag_dir / f"{frag['id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(frag, f, indent=4)
        print(f"[SymbolGen] Saved fragment: {frag['id']}")

def run_symbol_generator():
    # Lazy imports to avoid circular import with transformers/__init__ (HindsightTransformer).
    from model_manager import load_config
    from transformers.fractal_multidimensional_transformers import FractalTransformer

    config = load_config()
    child = config.get("current_child", "default_child")
    transformer = FractalTransformer()

    reflection = load_self_reflection(child)
    procedural_symbols = procedural_combinations()
    enriched = enrich_symbols(procedural_symbols, transformer)

    if "self_generated_symbols" not in reflection:
        reflection["self_generated_symbols"] = []
    reflection["self_generated_symbols"].extend(enriched)

    save_self_reflection(child, reflection)
    save_as_fragments(child, enriched)
    print(f"[SymbolGen] Generated {len(enriched)} procedural symbols.")

if __name__ == "__main__":
    run_symbol_generator()
