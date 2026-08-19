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
DEFAULT_MEANING_LINKS = 4       # bounded senses retained per surface word
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


def _text_memory_policy() -> Dict[str, Any]:
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
    try:
        meaning_links = min(MAX_SYMBOL_LINKS, max(1, int(raw.get("meaning_links", DEFAULT_MEANING_LINKS))))
    except (TypeError, ValueError):
        meaning_links = DEFAULT_MEANING_LINKS
    try:
        meaning_slack = min(1.0, max(0.0, float(raw.get("meaning_slack", 0.22))))
    except (TypeError, ValueError):
        meaning_slack = 0.22
    return {
        "vocab_limit": vocab_limit,
        "link_batch_size": link_batch_size,
        "meaning_links": meaning_links,
        "meaning_slack": meaning_slack,
    }


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


def _vocab_source_category(source: Optional[str], tags: Optional[List[str]]) -> str:
    """Collapse detailed provenance into stable, monitor-friendly categories."""
    source_name = str(source or "").strip().lower()
    tag_set = {str(tag).strip().lower() for tag in tags or [] if tag}
    if "discord" in tag_set or source_name.startswith("discord"):
        return "discord"
    if source_name.startswith("self_read:"):
        source_key = source_name.split(":", 1)[1].strip()
        return f"self_read_{source_key}" if source_key else "self_read_other"
    if "self_read" in tag_set:
        if tag_set & {"code", "self_code", "project_source"}:
            return "self_read_code"
        if tag_set & {"book", "books", "book_library"}:
            return "self_read_books"
        return "self_read_other"
    return source_name.replace(":", "_") or "unspecified"


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
    source_category = _vocab_source_category(source, tag_list)

    for word in tokens:
        entry = vocab.setdefault(word, {"count": 0, "last_seen": now, "emotion_samples": 0})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_seen"] = now
        source_counts = entry.setdefault("sources", {})
        if isinstance(source_counts, dict):
            source_counts[source_category] = int(source_counts.get(source_category, 0)) + 1

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


def diagnose_text_alignment(native_text: str, english_text: str) -> Dict[str, Any]:
    """Find unambiguous replacements while preserving shared native context."""
    native_tokens = [
        token.strip() for token in str(native_text or "").split() if token.strip()
    ]
    gloss_tokens = [
        token.strip() for token in str(english_text or "").split() if token.strip()
    ]
    english_words = tokenize_text(english_text)
    candidate_pairs: List[Dict[str, str]] = []
    replacement_rejections = []
    unchanged_context_count = 0
    diagnostic = {
        "native_text": str(native_text or ""),
        "english_text": str(english_text or ""),
        "native_tokens": native_tokens,
        "gloss_tokens": gloss_tokens,
        "english_words": english_words,
        "native_token_count": len(native_tokens),
        "gloss_token_count": len(gloss_tokens),
        "english_word_count": len(english_words),
    }
    if not native_tokens:
        reason = "empty_native"
    elif not gloss_tokens:
        reason = "empty_human_guess"
    elif len(native_tokens) != len(gloss_tokens):
        reason = "sequence_length_mismatch"
    else:
        for index, (native_token, gloss_token) in enumerate(
            zip(native_tokens, gloss_tokens)
        ):
            if native_token == gloss_token:
                unchanged_context_count += 1
                continue
            replacement_words = tokenize_text(gloss_token)
            if len(replacement_words) == 1:
                candidate_pairs.append(
                    {
                        "native": native_token,
                        "english": replacement_words[0],
                    }
                )
            else:
                replacement_rejections.append(
                    {
                        "index": index,
                        "native": native_token,
                        "gloss": gloss_token,
                        "reason": (
                            "non_english_replacement"
                            if not replacement_words
                            else "ambiguous_replacement"
                        ),
                        "english_words": replacement_words,
                    }
                )
        if candidate_pairs:
            reason = "accepted_contextual"
        elif unchanged_context_count == len(native_tokens):
            reason = "no_changed_tokens"
        else:
            reason = "no_unambiguous_replacements"

    diagnostic.update(
        {
            "reason": reason,
            "accepted": reason == "accepted_contextual",
            "candidate_pairs": candidate_pairs,
            "unchanged_context_count": unchanged_context_count,
            "replacement_rejections": replacement_rejections,
        }
    )
    return diagnostic


def review_text_evidence(
    observations: List[Dict[str, Any]],
    alignments: List[tuple[str, str]],
    *,
    child: Optional[str] = None,
    source: str = "language_review",
) -> Dict[str, Any]:
    """Apply a bounded batch of chat observations and explicit alignments once."""
    vocab_state = load_text_vocab(child)
    vocab = vocab_state.get("vocab", {})
    now = _now_iso()
    observed_messages = 0
    pairs: List[Dict[str, str]] = []
    alignment_diagnostics: List[Dict[str, Any]] = []

    def observe(text: str, tags: Optional[List[str]] = None) -> List[str]:
        nonlocal observed_messages
        words = tokenize_text(text)
        if not words:
            return []
        observed_messages += 1
        tag_list = [str(tag) for tag in tags or [] if tag]
        source_category = _vocab_source_category(source, tag_list)
        for word in words:
            entry = vocab.setdefault(
                word, {"count": 0, "last_seen": now, "emotion_samples": 0}
            )
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["last_seen"] = now
            source_counts = entry.setdefault("sources", {})
            if isinstance(source_counts, dict):
                source_counts[source_category] = int(source_counts.get(source_category, 0)) + 1
            if tag_list:
                existing = [str(tag) for tag in entry.get("tags", []) if tag]
                for tag in tag_list:
                    if tag not in existing:
                        existing.append(tag)
                entry["tags"] = existing[-6:]
            vocab[word] = entry
        return words

    for observation in observations or []:
        if not isinstance(observation, dict):
            continue
        observe(str(observation.get("text") or ""), observation.get("tags"))

    for native_text, english_text in alignments or []:
        diagnostic = diagnose_text_alignment(native_text, english_text)
        english_words = observe(
            str(english_text or ""), ["discord", "history", "symbolic_alignment"]
        )
        diagnostic["observed_english_words"] = list(english_words)
        alignment_diagnostics.append(diagnostic)
        if not diagnostic["accepted"]:
            continue
        for candidate_pair in diagnostic["candidate_pairs"]:
            native_token = candidate_pair["native"]
            english_word = candidate_pair["english"]
            entry = vocab.get(english_word)
            if not isinstance(entry, dict):
                continue
            symbol_counts = entry.setdefault("symbols", {})
            symbol_counts[native_token] = int(symbol_counts.get(native_token, 0)) + 1
            top = sorted(
                symbol_counts.items(), key=lambda item: (-int(item[1]), item[0])
            )[:MAX_SYMBOL_LINKS]
            entry["symbols"] = {symbol: count for symbol, count in top}
            vocab[english_word] = entry
            pairs.append({"native": native_token, "english": english_word})

    if observed_messages:
        vocab_state["vocab"] = vocab
        vocab_state["updated"] = now
        save_text_vocab(child, vocab_state, source=source)
    return {
        "updated": observed_messages > 0,
        "observed_messages": observed_messages,
        "alignment_candidates": len(alignment_diagnostics),
        "accepted_alignment_candidates": sum(
            1 for diagnostic in alignment_diagnostics if diagnostic["accepted"]
        ),
        "alignment_rejections": [
            diagnostic for diagnostic in alignment_diagnostics
            if not diagnostic["accepted"]
        ],
        "alignment_diagnostics": alignment_diagnostics,
        "pairs": pairs,
    }


def observe_text_symbol_alignment(
    native_text: str,
    english_text: str,
    *,
    child: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: str = "symbolic_alignment",
) -> Dict[str, Any]:
    """Retain a one-to-one native/English alignment as revisable evidence."""
    result = review_text_evidence(
        [],
        [(native_text, english_text)],
        child=child,
        source=source,
    )
    if result["observed_messages"] and not result["pairs"]:
        result["reason"] = "unaligned_lengths"
    else:
        result["reason"] = "aligned" if result["pairs"] else "empty_alignment"
    return result


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


def _word_evidence_revision(meta: Any) -> str:
    evidence = meta if isinstance(meta, dict) else {}
    payload = {
        "count": int(evidence.get("count", 0) or 0),
        "tags": sorted(str(tag) for tag in evidence.get("tags", []) if tag),
        "symbols": {
            str(symbol): int(count)
            for symbol, count in (evidence.get("symbols") or {}).items()
            if symbol
        } if isinstance(evidence.get("symbols"), dict) else {},
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_text_symbol_links(
    child: Optional[str] = None,
    *,
    top_words: Optional[int] = None,
    similarity_threshold: float = 0.42,
    mapping_batch: Optional[int] = None,
    revisit_existing: int = 0,
) -> bool:
    """Incrementally map retained words and bounded existing links to symbols."""
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
    links_by_word: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for link in prior.get("links") or []:
        if not isinstance(link, dict) or not link.get("word") or not link.get("symbol"):
            continue
        word = str(link["word"])
        if word in vocab_words:
            links_by_word.setdefault(word, {})[str(link["symbol"])] = link
    evaluated = {
        str(word): value for word, value in evaluated.items()
        if str(word) in vocab_words
    }

    ranked_words = sorted(
        vocab.items(), key=lambda kv: (-int(kv[1].get("count", 0)), kv[1].get("last_seen", ""))
    )
    if top_words is not None:
        ranked_words = ranked_words[:max(0, int(top_words))]
    pending_words = []
    for word, meta in ranked_words:
        evidence_revision = _word_evidence_revision(meta)
        prior_evaluation = evaluated.get(word)
        if prior_evaluation is True:
            # Migrate the legacy boolean ledger without forcing a full-vocab pass.
            evaluated[word] = {
                "evidence_revision": evidence_revision,
                "evaluated_at": prior.get("generated") or _now_iso(),
                "migrated": True,
            }
            continue
        prior_revision = (
            str(prior_evaluation.get("evidence_revision") or "")
            if isinstance(prior_evaluation, dict)
            else ""
        )
        if prior_revision != evidence_revision:
            pending_words.append((word, meta))

    batch_limit = _text_memory_policy()["link_batch_size"] if mapping_batch is None else max(1, int(mapping_batch))
    batch = pending_words[:batch_limit]
    new_batch_count = len(batch)
    revisit_batch_count = 0
    if revisit_existing > 0 and len(batch) < batch_limit:
        pending_names = {word for word, _meta in pending_words}
        links_by_priority = sorted(
            (link for meanings in links_by_word.values() for link in meanings.values()),
            key=lambda link: (
                float(link.get("similarity", 0.0) or 0.0),
                float(link.get("symbol_confidence", 0.0) or 0.0),
                int(link.get("count", 0) or 0),
                str(link.get("word") or ""),
            ),
        )
        revisit_names = list(dict.fromkeys(
            str(link.get("word"))
            for link in links_by_priority
            if link.get("word") in vocab and str(link.get("word")) not in pending_names
        ))[:max(0, int(revisit_existing))]
        room = max(0, batch_limit - len(batch))
        revisited = revisit_names[:room]
        batch.extend((word, vocab[word]) for word in revisited)
        revisit_batch_count = len(revisited)

    for word, meta in batch:
        lang = guess_language_code(word)
        w_emb = _get_embedder().embed_text(word, language=lang)
        candidates = []
        symbol_evidence = meta.get("symbols") if isinstance(meta, dict) else {}
        symbol_evidence = symbol_evidence if isinstance(symbol_evidence, dict) else {}
        for sid, emb, symbol_word, symbol_confidence in sym_entries:
            sim = _get_embedder().cosine(w_emb, emb)
            evidence_count = int(
                symbol_evidence.get(sid, symbol_evidence.get(symbol_word, 0)) or 0
            )
            evidence_bonus = min(0.35, 0.12 * evidence_count)
            score = sim + evidence_bonus
            candidates.append((score, sim, str(sid), symbol_word, symbol_confidence, evidence_count))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        best_score = candidates[0][0] if candidates else 0.0
        policy = _text_memory_policy()
        retained = [
            item for item in candidates
            if item[0] >= similarity_threshold
            and (item[0] >= best_score - float(policy["meaning_slack"]) or item[5] > 0)
        ][:int(policy["meaning_links"])]
        old_meanings = links_by_word.get(word, {})
        new_meanings: Dict[str, Dict[str, Any]] = {}
        reviewed_at = _now_iso()
        sources = dict(meta.get("sources") or {}) if isinstance(meta, dict) and isinstance(meta.get("sources"), dict) else {}
        contexts = [str(tag) for tag in (meta.get("tags") or []) if tag][:6] if isinstance(meta, dict) else []
        for score, sim, sid, symbol_word, symbol_confidence, evidence_count in retained:
            old = old_meanings.get(sid, {})
            prior_strength = float(old.get("strength", old.get("mapping_score", score)) or 0.0)
            observed_strength = max(0.0, min(1.0, score))
            strength = observed_strength if not old else (prior_strength * 0.85) + (observed_strength * 0.15)
            reinforcement_count = int(old.get("reinforcement_count", old.get("usage_count", 0)) or 0)
            last_reinforced = old.get("last_reinforced") or old.get("last_seen")
            if evidence_count > int(old.get("evidence_count", 0) or 0):
                reinforcement_count += 1
                last_reinforced = reviewed_at
            new_meanings[sid] = {
                "word": word,
                "count": int(meta.get("count", 0)),
                "last_seen": meta.get("last_seen"),
                "symbol": sid,
                "symbol_word": symbol_word,
                "symbol_confidence": symbol_confidence,
                "similarity": round(sim, 4),
                "mapping_score": round(score, 4),
                "strength": round(strength, 4),
                "usage_count": int(evidence_count),
                "reinforcement_count": reinforcement_count,
                "last_reinforced": last_reinforced,
                "last_reviewed": reviewed_at,
                "created_at": old.get("created_at") or reviewed_at,
                "sources": sources,
                "contexts": contexts,
                "decay_policy": "event_driven_review",
                "evidence_count": int(evidence_count),
            }
        if new_meanings:
            links_by_word[word] = new_meanings
        else:
            links_by_word.pop(word, None)
        evaluated[word] = {
            "evidence_revision": _word_evidence_revision(meta),
            "evaluated_at": _now_iso(),
        }

    rank = {word: index for index, (word, _meta) in enumerate(ranked_words)}
    links = sorted(
        (link for meanings in links_by_word.values() for link in meanings.values()),
        key=lambda link: (rank.get(str(link.get("word")), len(rank)), -float(link.get("strength", 0.0))),
    )
    remaining_words = pending_words[new_batch_count:]
    remaining = len(remaining_words)
    queue_by_source: Dict[str, int] = {}
    for _word, meta in remaining_words:
        sources = meta.get("sources") if isinstance(meta, dict) else {}
        sources = sources if isinstance(sources, dict) else {}
        ranked_sources = sorted(
            ((str(name), int(count or 0)) for name, count in sources.items()),
            key=lambda item: (-item[1], item[0]),
        )
        category = ranked_sources[0][0] if ranked_sources else "legacy_unspecified"
        if len(ranked_sources) > 1 and ranked_sources[0][1] == ranked_sources[1][1]:
            category = "mixed"
        queue_by_source[category] = int(queue_by_source.get(category, 0)) + 1
    if new_batch_count and revisit_batch_count:
        batch_mode = "new_and_revisit"
    elif new_batch_count:
        batch_mode = "new"
    elif revisit_batch_count:
        batch_mode = "revisit"
    else:
        batch_mode = "idle"
    payload = {
        "schema_version": 2,
        "meaning_model": "one_word_to_ranked_links",
        "generated": _now_iso(),
        "symbol_source_revision": symbol_revision,
        "evaluated": evaluated,
        "evaluated_count": len(evaluated),
        "remaining": remaining,
        "queue_by_source": dict(sorted(queue_by_source.items())),
        "last_batch": {
            "mode": batch_mode,
            "new_mappings": new_batch_count,
            "revisited_mappings": revisit_batch_count,
        },
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
    "diagnose_text_alignment",
    "review_text_evidence",
    "observe_text_symbol_alignment",
    "create_text_fragment",
    "record_text_observation",
    "build_text_symbol_links",
]
