import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from embedding_stack import MultimodalEmbedder, guess_language_code

TEXT_VOCAB_LIMIT = 800          # keep only the top-K words to stay RAM-light
MAX_FRAGMENT_BODY = 1200        # cap stored text per fragment
MAX_FRAGMENT_SUMMARY = 240      # short preview for scans
MAX_SYMBOL_LINKS = 6            # per-word symbol co-occurrence cap

_EMBEDDER: Optional[MultimodalEmbedder] = None


def _get_embedder() -> MultimodalEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = MultimodalEmbedder(dim=128)
    return _EMBEDDER


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_child(child: Optional[str]) -> str:
    if child:
        return str(child)
    cfg_path = Path("config.json")
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("current_child"):
                return str(data["current_child"])
        except Exception:
            pass
    return "Inazuma_Yagami"


def _memory_root(child: Optional[str]) -> Path:
    return Path("AI_Children") / _safe_child(child) / "memory"


def _safe_text(value: Any, limit: int = MAX_FRAGMENT_BODY) -> str:
    text = "" if value is None else str(value)
    cleaned = "".join(ch for ch in text if ch.isprintable())
    return cleaned[:limit]


def tokenize_text(text: str) -> List[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", text or "") if tok]


def load_text_vocab(child: Optional[str] = None) -> Dict[str, Any]:
    path = _memory_root(child) / "text_vocab.json"
    if not path.exists():
        return {"vocab": {}, "updated": _now_iso()}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                data.setdefault("vocab", {})
                return data
    except Exception:
        pass
    return {"vocab": {}, "updated": _now_iso()}


def _trim_vocab(vocab: Dict[str, Any], limit: int) -> Dict[str, Any]:
    items = sorted(
        vocab.items(),
        key=lambda kv: (-int(kv[1].get("count", 0)), kv[1].get("last_seen", "")),
    )
    return {w: data for w, data in items[:limit]}


def save_text_vocab(child: Optional[str], data: Dict[str, Any], limit: int = TEXT_VOCAB_LIMIT) -> None:
    vocab = data.get("vocab", {})
    trimmed = _trim_vocab(vocab, limit)
    payload = {"vocab": trimmed, "updated": data.get("updated", _now_iso())}
    path = _memory_root(child) / "text_vocab.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_text_vocab(
    text: str,
    *,
    child: Optional[str] = None,
    tags: Optional[List[str]] = None,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    limit: int = TEXT_VOCAB_LIMIT,
) -> bool:
    tokens = tokenize_text(text)
    if not tokens:
        return False

    vocab_state = load_text_vocab(child)
    vocab = vocab_state.get("vocab", {})
    now = _now_iso()
    tag_list = [str(t) for t in tags or [] if t]

    for word in tokens:
        entry = vocab.setdefault(word, {"count": 0, "last_seen": now, "emotion_samples": 0})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_seen"] = now

        if tag_list:
            existing = [str(t) for t in entry.get("tags", []) if t]
            for t in tag_list:
                if t not in existing:
                    existing.append(t)
            entry["tags"] = existing[-6:]

        if emotions:
            emo_store = entry.setdefault("emotions", {})
            samples = int(entry.get("emotion_samples", 0))
            samples += 1
            for k, v in emotions.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                prev = emo_store.get(k, 0.0)
                emo_store[k] = round(((prev * (samples - 1)) + val) / samples, 4)
            entry["emotion_samples"] = samples

        if symbols:
            sym_store = entry.setdefault("symbols", {})
            for sid in symbols:
                if not sid:
                    continue
                sym_store[str(sid)] = int(sym_store.get(str(sid), 0)) + 1
            top = sorted(sym_store.items(), key=lambda kv: -kv[1])[:MAX_SYMBOL_LINKS]
            entry["symbols"] = {k: v for k, v in top}

        vocab[word] = entry

    vocab_state["vocab"] = vocab
    vocab_state["updated"] = now
    save_text_vocab(child, vocab_state, limit=limit)
    return True


def create_text_fragment(
    text: str,
    *,
    source: str = "",
    child: Optional[str] = None,
    tags: Optional[List[str]] = None,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    importance: Optional[float] = None,
) -> Dict[str, Any]:
    child_name = _safe_child(child)
    text_body = _safe_text(text)
    if not text_body:
        return {}

    frag_id = f"frag_text_{uuid.uuid4().hex[:10]}"
    frag_tags = ["text"] + [t for t in tags or [] if t]
    frag_tags = list(dict.fromkeys(frag_tags))  # preserve order, de-dupe

    frag = {
        "id": frag_id,
        "modality": "text",
        "summary": text_body[:MAX_FRAGMENT_SUMMARY],
        "text": text_body,
        "source": source,
        "tags": frag_tags,
        "timestamp": _now_iso(),
        "length": len(text_body),
    }
    if emotions:
        frag["emotions"] = emotions
    if symbols:
        frag["symbols"] = symbols
    if importance is not None:
        frag["importance"] = importance

    frag_path = _memory_root(child_name) / "fragments" / f"{frag_id}.json"
    frag_path.parent.mkdir(parents=True, exist_ok=True)
    frag_path.write_text(json.dumps(frag, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    update_text_vocab(text_body, child=child_name, tags=frag_tags, emotions=emotions, symbols=symbols, limit=TEXT_VOCAB_LIMIT)
    return frag


def record_text_observation(
    text: str,
    *,
    source: str = "",
    child: Optional[str] = None,
    tags: Optional[List[str]] = None,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    importance: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    try:
        return create_text_fragment(
            text,
            source=source,
            child=child,
            tags=tags,
            emotions=emotions,
            symbols=symbols,
            importance=importance,
        )
    except Exception:
        return None


def build_text_symbol_links(
    child: Optional[str] = None,
    *,
    top_words: int = 120,
    similarity_threshold: float = 0.42,
) -> bool:
    """
    Map frequent text words to nearest-known symbols (based on vocab embeddings).
    Produces memory/text_vocab_links.json for inspection.
    """
    child_name = _safe_child(child)
    vocab_state = load_text_vocab(child_name)
    vocab = vocab_state.get("vocab", {})
    if not vocab:
        return False

    sym_path = _memory_root(child_name) / "symbol_to_token.json"
    if not sym_path.exists():
        return False
    try:
        sym_vocab = json.loads(sym_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    sym_entries = []
    for sid, entry in sym_vocab.items():
        if not isinstance(entry, dict):
            continue
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        emb = entry.get("embedding")
        if not isinstance(emb, list) or not emb:
            lang = entry.get("language") or guess_language_code(word)
            emb = _get_embedder().embed_text(word, language=lang)
        sym_entries.append((sid, emb, word, entry.get("confidence")))

    if not sym_entries:
        return False

    ranked_words = sorted(
        vocab.items(), key=lambda kv: (-int(kv[1].get("count", 0)), kv[1].get("last_seen", ""))
    )[:top_words]

    links = []
    for word, meta in ranked_words:
        lang = guess_language_code(word)
        w_emb = _get_embedder().embed_text(word, language=lang)
        best = None
        best_sim = 0.0
        best_word = None
        best_conf = None
        for sid, emb, s_word, s_conf in sym_entries:
            sim = _get_embedder().cosine(w_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best = sid
                best_word = s_word
                best_conf = s_conf
        if best and best_sim >= similarity_threshold:
            links.append(
                {
                    "word": word,
                    "count": int(meta.get("count", 0)),
                    "last_seen": meta.get("last_seen"),
                    "symbol": best,
                    "symbol_word": best_word,
                    "symbol_confidence": best_conf,
                    "similarity": round(best_sim, 4),
                }
            )

    out_path = _memory_root(child_name) / "text_vocab_links.json"
    out_path.write_text(
        json.dumps({"generated": _now_iso(), "links": links}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


__all__ = [
    "tokenize_text",
    "load_text_vocab",
    "save_text_vocab",
    "update_text_vocab",
    "create_text_fragment",
    "record_text_observation",
    "build_text_symbol_links",
]
