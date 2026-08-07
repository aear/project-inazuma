import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from embedding_stack import MultimodalEmbedder, guess_language_code
from model_manager import increment_inastate_metric, set_inastate_metric

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

DEFAULT_TEXT_VOCAB_LIMIT = 25_000  # realistic active vocabulary; configurable
DEFAULT_LINK_BATCH_SIZE = 500   # bounded work per meaning-map pass
TEXT_VOCAB_LIMIT = DEFAULT_TEXT_VOCAB_LIMIT  # compatibility alias
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


def _text_memory_policy() -> Dict[str, int]:
    raw: Dict[str, Any] = {}
    try:
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        candidate = config.get("text_memory_policy") if isinstance(config, dict) else None
        raw = candidate if isinstance(candidate, dict) else {}
    except Exception:
        raw = {}
    try:
        vocab_limit = max(1_000, int(raw.get("vocab_limit", DEFAULT_TEXT_VOCAB_LIMIT)))
    except (TypeError, ValueError):
        vocab_limit = DEFAULT_TEXT_VOCAB_LIMIT
    try:
        link_batch_size = max(1, int(raw.get("link_batch_size", DEFAULT_LINK_BATCH_SIZE)))
    except (TypeError, ValueError):
        link_batch_size = DEFAULT_LINK_BATCH_SIZE
    return {"vocab_limit": vocab_limit, "link_batch_size": link_batch_size}


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


@contextmanager
def _json_lock(path: Path):
    if not fcntl:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _atomic_write_json(path: Path, payload: Any, *, indent: int = 2, ensure_ascii: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=indent, ensure_ascii=ensure_ascii)
            tmp.write("\n")
    finally:
        os.replace(tmp_path, path)


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


def save_text_vocab(
    child: Optional[str],
    data: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    source: Optional[str] = None,
) -> None:
    vocab = data.get("vocab", {})
    effective_limit = _text_memory_policy()["vocab_limit"] if limit is None else max(1, int(limit))
    trimmed = _trim_vocab(vocab, effective_limit)
    payload = {"vocab": trimmed, "updated": data.get("updated", _now_iso())}
    path = _memory_root(child) / "text_vocab.json"
    with _json_lock(path):
        _atomic_write_json(path, payload, indent=2, ensure_ascii=False)
    try:
        increment_inastate_metric("vocab_updates")
        set_inastate_metric("last_vocab_update_source", source or "unspecified")
    except Exception:
        pass


def update_text_vocab(
    text: str,
    *,
    child: Optional[str] = None,
    tags: Optional[List[str]] = None,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    limit: Optional[int] = None,
    source: Optional[str] = None,
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
    save_text_vocab(child, vocab_state, limit=limit, source=source)
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

    vocab_source = source or "text_fragment"
    update_text_vocab(
        text_body,
        child=child_name,
        tags=frag_tags,
        emotions=emotions,
        symbols=symbols,
        limit=None,
        source=vocab_source,
    )
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
    top_words: Optional[int] = None,
    similarity_threshold: float = 0.42,
    mapping_batch: Optional[int] = None,
) -> bool:
    """Incrementally map every retained vocabulary word to known symbols."""
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
    symbol_revision = hashlib.sha1(
        json.dumps(
            sorted((str(sid), str(word), confidence, emb) for sid, emb, word, confidence in sym_entries),
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    out_path = _memory_root(child_name) / "text_vocab_links.json"
    try:
        prior = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    except Exception:
        prior = {}
    prior = prior if isinstance(prior, dict) else {}
    evaluated = prior.get("evaluated") if isinstance(prior.get("evaluated"), dict) else {}
    if str(prior.get("symbol_source_revision") or "") != symbol_revision:
        evaluated = {}
    vocab_words = set(str(word) for word in vocab)
    links_by_word = {
        str(link.get("word")): link for link in (prior.get("links") or [])
        if isinstance(link, dict) and link.get("word") and str(link.get("word")) in vocab_words
    }
    evaluated = {str(word): True for word in evaluated if str(word) in vocab_words}

    ranked_words = sorted(
        vocab.items(), key=lambda kv: (-int(kv[1].get("count", 0)), kv[1].get("last_seen", ""))
    )
    if top_words is not None:
        ranked_words = ranked_words[:max(0, int(top_words))]
    pending_words = [(word, meta) for word, meta in ranked_words if word not in evaluated]
    batch_limit = _text_memory_policy()["link_batch_size"] if mapping_batch is None else max(1, int(mapping_batch))
    batch = pending_words[:batch_limit]

    for word, meta in batch:
        lang = guess_language_code(word)
        w_emb = _get_embedder().embed_text(word, language=lang)
        best = None
        best_sim = 0.0
        best_word = None
        best_conf = None
        for sid, emb, symbol_word, symbol_confidence in sym_entries:
            sim = _get_embedder().cosine(w_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best = sid
                best_word = symbol_word
                best_conf = symbol_confidence
        if best and best_sim >= similarity_threshold:
            links_by_word[word] = {
                "word": word,
                "count": int(meta.get("count", 0)),
                "last_seen": meta.get("last_seen"),
                "symbol": best,
                "symbol_word": best_word,
                "symbol_confidence": best_conf,
                "similarity": round(best_sim, 4),
            }
        else:
            links_by_word.pop(word, None)
        evaluated[word] = True

    rank = {word: index for index, (word, _meta) in enumerate(ranked_words)}
    links = sorted(links_by_word.values(), key=lambda link: rank.get(str(link.get("word")), len(rank)))
    remaining = max(0, len(pending_words) - len(batch))
    payload = {
        "generated": _now_iso(),
        "symbol_source_revision": symbol_revision,
        "evaluated": evaluated,
        "evaluated_count": len(evaluated),
        "remaining": remaining,
        "complete": remaining == 0,
        "links": links,
    }
    with _json_lock(out_path):
        _atomic_write_json(out_path, payload, indent=2, ensure_ascii=False)
    try:
        increment_inastate_metric("link_pass_runs")
    except Exception:
        pass
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
