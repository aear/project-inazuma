"""
Typed neural graph for Ina: sound → token → word → symbol.
Builds a compact, typed graph to reduce symbol collapse and to ground new
English learning without replacing existing memory systems.
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from embedding_stack import MultimodalEmbedder, guess_language_code
from gui_hook import log_to_statusbox
from model_manager import load_config, update_inastate

GRAPH_FILENAME = "typed_neural_graph.json"
DEFAULT_BURST = 40  # number of expression fragments to fold in per run


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except Exception:
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _reduce_vector(vec: Iterable[float], target: int = 16) -> List[float]:
    trimmed = [float(v) for v in (vec or [])][:target]
    if len(trimmed) < target:
        trimmed.extend([0.0] * (target - len(trimmed)))
    return trimmed


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _split_subtokens(word: str) -> List[str]:
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", word or "") if tok]
    if tokens:
        return tokens
    compact = re.sub(r"\W+", "", (word or "").lower())
    if len(compact) > 3:
        return [compact[:3], compact[-3:]]
    return [compact] if compact else []


def _weight_from_uses(confidence: float, uses: int, *, bias: float = 0.35) -> float:
    conf = max(0.0, min(1.0, confidence))
    strength = math.log1p(max(uses, 0)) / 8.0
    return round(min(1.0, bias + (0.5 * conf) + min(0.45, strength)), 4)


class TypedGraphBuilder:
    def __init__(self, child: str, base_path: Optional[Path] = None):
        self.child = child
        self.base = Path(base_path) if base_path else Path("AI_Children")
        self.memory_root = self.base / child / "memory"
        self.graph_path = self.memory_root / "neural" / GRAPH_FILENAME
        self.embedder = MultimodalEmbedder(dim=64)

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {
            "child": child,
            "generated_at": _now_iso(),
            "progress": {"expression_cursor": None, "processed_fragments": 0},
            "counts": {},
            "sources": {},
        }

    # --- graph helpers -------------------------------------------------
    def load_existing(self) -> None:
        data = _load_json(self.graph_path)
        if not isinstance(data, dict):
            return
        raw_nodes = data.get("nodes") or {}
        if isinstance(raw_nodes, list):
            for node in raw_nodes:
                if isinstance(node, dict) and node.get("id"):
                    self.nodes[node["id"]] = node
        elif isinstance(raw_nodes, dict):
            self.nodes.update({k: v for k, v in raw_nodes.items() if isinstance(v, dict)})

        raw_edges = data.get("edges") or {}
        if isinstance(raw_edges, list):
            for edge in raw_edges:
                if not isinstance(edge, dict):
                    continue
                key = self._edge_key(edge.get("source"), edge.get("target"), edge.get("relation"))
                if key:
                    self.edges[key] = edge
        elif isinstance(raw_edges, dict):
            for key, edge in raw_edges.items():
                if isinstance(edge, dict):
                    self.edges[key] = edge

        if isinstance(data.get("metadata"), dict):
            self.metadata.update(data["metadata"])

    def save(self) -> None:
        payload = {
            "metadata": {
                **self.metadata,
                "generated_at": _now_iso(),
                "counts": self._counts(),
                "sources": self._source_counts(),
            },
            "nodes": self.nodes,
            "edges": self.edges,
        }
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with self.graph_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.get("type", "unknown")] = counts.get(node.get("type", "unknown"), 0) + 1
        counts["edges"] = len(self.edges)
        return counts

    def _source_counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for node in self.nodes.values():
            for src in node.get("sources", []):
                tally[src] = tally.get(src, 0) + 1
        for edge in self.edges.values():
            for src in edge.get("sources", []):
                tally[src] = tally.get(src, 0) + 1
        return tally

    @staticmethod
    def _edge_key(source: Optional[str], target: Optional[str], relation: Optional[str]) -> Optional[str]:
        if not source or not target or not relation:
            return None
        return f"{relation}|{source}|{target}"

    def _upsert_node(self, node_id: str, node_type: str, *, label: Optional[str] = None, **fields: Any) -> None:
        node = self.nodes.get(node_id, {"id": node_id, "type": node_type})
        if label:
            node["label"] = label
        sources = set(node.get("sources", []))
        if fields.get("source"):
            sources.add(fields["source"])
            fields.pop("source", None)
        if sources:
            node["sources"] = sorted(sources)

        for key, val in fields.items():
            if val is None:
                continue
            if key in ("uses", "count"):
                node[key] = int(val)
            else:
                node[key] = val
        self.nodes[node_id] = node

    def _upsert_edge(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        weight: float,
        evidence: int = 1,
        last_seen: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> None:
        key = self._edge_key(source, target, relation)
        if not key:
            return
        edge = self.edges.get(
            key,
            {"source": source, "target": target, "relation": relation, "weight": 0.0, "evidence": 0},
        )

        edge["weight"] = round(max(edge.get("weight", 0.0), min(1.0, max(0.0, weight))), 4)
        edge["evidence"] = int(edge.get("evidence", 0)) + max(1, evidence)
        if last_seen:
            edge["last_seen"] = last_seen

        srcs = set(edge.get("sources", []))
        if source_tag:
            srcs.add(source_tag)
        if srcs:
            edge["sources"] = sorted(srcs)

        self.edges[key] = edge

    # --- data ingestion ------------------------------------------------
    def ingest_sound_symbols(self) -> None:
        path = self.memory_root / "sound_symbol_map.json"
        raw = _load_json(path) or {}
        symbols = raw.get("symbols") if isinstance(raw, dict) else {}
        if not isinstance(symbols, dict):
            symbols = {}

        for sid, entry in symbols.items():
            if not isinstance(entry, dict):
                continue
            node_id = f"sound:{sid}"
            uses = int(entry.get("uses", 0))
            emb = _reduce_vector(
                self.embedder.embed_audio_frames([entry.get("centroid", [])], texture=entry.get("texture"))
            )
            self._upsert_node(
                node_id,
                "sound",
                label=sid,
                uses=uses,
                last_seen=entry.get("last_seen"),
                embedding_hint=emb,
                source="sound_symbol_map",
            )

    def ingest_symbol_words(self) -> None:
        path = self.memory_root / "symbol_words.json"
        data = _load_json(path) or {}
        words = data.get("words") if isinstance(data, dict) else None
        if not isinstance(words, list):
            return

        for entry in words:
            if not isinstance(entry, dict):
                continue
            sym_id = entry.get("symbol_word_id")
            if not sym_id:
                continue
            node_id = f"symbol:{sym_id}"
            self._upsert_node(
                node_id,
                "symbol",
                label=entry.get("symbol") or sym_id,
                uses=entry.get("count", 0),
                confidence=entry.get("confidence"),
                summary=entry.get("summary"),
                source="symbol_words",
            )
            generated_word = (entry.get("generated_word") or "").strip().lower()
            if generated_word and generated_word != "unknown":
                wid = f"word:{generated_word}"
                self._upsert_node(
                    wid,
                    "word",
                    label=generated_word,
                    language=guess_language_code(generated_word),
                    embedding_hint=_reduce_vector(self.embedder.embed_text(generated_word)),
                    source="symbol_words",
                )
                weight = _weight_from_uses(entry.get("confidence", 0.2), entry.get("usage_count", 0), bias=0.2)
                self._upsert_edge(node_id, wid, "symbol->word", weight=weight, evidence=1, source_tag="symbol_words")

    def ingest_symbol_vocab(self) -> None:
        path = self.memory_root / "symbol_to_token.json"
        vocab = _load_json(path) or {}
        if not isinstance(vocab, dict):
            return

        for sym_id, entry in vocab.items():
            if not isinstance(entry, dict):
                continue
            word = (entry.get("word") or "").strip()
            if not word:
                continue

            word_lower = word.lower()
            lang = entry.get("language") or guess_language_code(word)
            uses = int(entry.get("uses", 0))
            conf = float(entry.get("confidence", 0.2))

            word_id = f"word:{word_lower}"
            self._upsert_node(
                word_id,
                "word",
                label=word,
                language=lang,
                embedding_hint=_reduce_vector(self.embedder.embed_text(word, language=lang)),
                source="symbol_to_token",
            )

            # Tokens derived from the word to prevent collapse at the subword level.
            for tok in _split_subtokens(word):
                tok_id = f"token:{tok}"
                self._upsert_node(tok_id, "token", label=tok, source="symbol_to_token")
                tok_weight = _weight_from_uses(conf, uses, bias=0.25)
                self._upsert_edge(word_id, tok_id, "word->token", weight=tok_weight, source_tag="symbol_to_token")

            sound_id = f"sound:{sym_id}"
            edge_weight = _weight_from_uses(conf, uses, bias=0.3)
            self._upsert_edge(sound_id, word_id, "sound->word", weight=edge_weight, source_tag="symbol_to_token")

    def ingest_text_vocab(self) -> None:
        path = self.memory_root / "text_vocab.json"
        vocab_state = _load_json(path) or {}
        vocab = vocab_state.get("vocab") if isinstance(vocab_state, dict) else None
        if not isinstance(vocab, dict):
            return

        for word, meta in vocab.items():
            word_lower = (word or "").strip().lower()
            if not word_lower:
                continue
            word_id = f"word:{word_lower}"
            lang = guess_language_code(word_lower)
            self._upsert_node(
                word_id,
                "word",
                label=word_lower,
                language=lang,
                embedding_hint=_reduce_vector(self.embedder.embed_text(word_lower, language=lang)),
                uses=meta.get("count"),
                source="text_vocab",
            )
            for tok in _split_subtokens(word_lower):
                tok_id = f"token:{tok}"
                self._upsert_node(tok_id, "token", label=tok, source="text_vocab")
                weight = _weight_from_uses(meta.get("count", 1), meta.get("count", 1), bias=0.2)
                self._upsert_edge(word_id, tok_id, "word->token", weight=weight, source_tag="text_vocab")

    # --- burst processing ----------------------------------------------
    def _iter_expression_fragments(self) -> List[Tuple[Path, datetime]]:
        frag_dir = self.memory_root / "fragments"
        paths = list(frag_dir.glob("frag_expression_*.json"))
        timed: List[Tuple[Path, datetime]] = []
        for path in paths:
            ts = _parse_ts(path.stem.split("_")[-1])  # coarse fallback
            if ts is None:
                data = _load_json(path) or {}
                ts = _parse_ts(data.get("timestamp"))
            if ts:
                timed.append((path, ts))
        timed.sort(key=lambda pair: pair[1])
        return timed

    def fold_expression_fragments(self, limit: int = DEFAULT_BURST) -> int:
        progress = self.metadata.setdefault("progress", {})
        cursor_ts = _parse_ts(progress.get("expression_cursor"))
        processed = 0
        last_seen_ts = cursor_ts

        for path, ts in self._iter_expression_fragments():
            if cursor_ts and ts <= cursor_ts:
                continue
            if processed >= limit:
                break

            data = _load_json(path) or {}
            sound = data.get("sound_symbol")
            symbol_word = data.get("symbol_word_id")
            vocab_word = (data.get("vocab_word") or "").strip()
            vocab_conf = float(data.get("vocab_word_confidence") or 0.0)
            symbol_conf = float(data.get("symbol_word_confidence") or 0.0)

            last_seen_iso = ts.isoformat()
            if sound and symbol_word:
                self._upsert_edge(
                    f"sound:{sound}",
                    f"symbol:{symbol_word}",
                    "sound->symbol",
                    weight=min(1.0, 0.4 + (symbol_conf * 0.6)),
                    evidence=1,
                    last_seen=last_seen_iso,
                    source_tag=path.name,
                )

            if vocab_word:
                word_id = f"word:{vocab_word.lower()}"
                self._upsert_node(
                    word_id,
                    "word",
                    label=vocab_word,
                    language=guess_language_code(vocab_word),
                    embedding_hint=_reduce_vector(self.embedder.embed_text(vocab_word)),
                    source="fragments",
                )
                if sound:
                    self._upsert_edge(
                        f"sound:{sound}",
                        word_id,
                        "sound->word",
                        weight=min(1.0, 0.25 + vocab_conf * 0.5),
                        evidence=1,
                        last_seen=last_seen_iso,
                        source_tag=path.name,
                    )
                if symbol_word:
                    self._upsert_edge(
                        f"symbol:{symbol_word}",
                        word_id,
                        "symbol->word",
                        weight=min(1.0, 0.25 + symbol_conf * 0.5),
                        evidence=1,
                        last_seen=last_seen_iso,
                        source_tag=path.name,
                    )
                for tok in _split_subtokens(vocab_word):
                    tok_id = f"token:{tok}"
                    self._upsert_node(tok_id, "token", label=tok, source="fragments")
                    self._upsert_edge(
                        word_id,
                        tok_id,
                        "word->token",
                        weight=min(1.0, 0.2 + vocab_conf * 0.6),
                        evidence=1,
                        last_seen=last_seen_iso,
                        source_tag=path.name,
                    )

            processed += 1
            last_seen_ts = ts if (last_seen_ts is None or ts > last_seen_ts) else last_seen_ts

        if processed and last_seen_ts:
            progress["expression_cursor"] = last_seen_ts.isoformat()
            progress["processed_fragments"] = int(progress.get("processed_fragments", 0)) + processed
        return processed


def build_typed_neural_graph(child: Optional[str] = None, *, base_path: Optional[Path] = None, burst: int = DEFAULT_BURST) -> Dict[str, Any]:
    cfg = load_config()
    child_name = child or cfg.get("current_child", "Inazuma_Yagami")
    builder = TypedGraphBuilder(child_name, base_path=base_path)
    builder.load_existing()

    builder.ingest_sound_symbols()
    builder.ingest_symbol_words()
    builder.ingest_symbol_vocab()
    builder.ingest_text_vocab()

    processed = builder.fold_expression_fragments(limit=burst)
    builder.save()

    summary = {
        "nodes": len(builder.nodes),
        "edges": len(builder.edges),
        "burst_processed": processed,
        "last_run": _now_iso(),
    }
    update_inastate("typed_neural_graph_status", summary)
    log_to_statusbox(
        f"[TypedNeuralGraph] nodes={summary['nodes']} edges={summary['edges']} "
        f"burst_processed={summary['burst_processed']}"
    )
    return summary


if __name__ == "__main__":
    build_typed_neural_graph()
