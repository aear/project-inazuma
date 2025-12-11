"""OpenAI-compatible shim to let LM Studio chat with Ina's cognition."""

from __future__ import annotations

import argparse
import json
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experience_logger import ExperienceLogger
from language_processing import (
    describe_word_grounding,
    load_symbol_to_token,
)
from live_experience_bridge import LiveExperienceBridge
from memory_graph import build_experience_graph
from model_manager import load_config, seed_self_question


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


class LMStudioAdapter:
    """Translate LM Studio chat prompts into Ina's grounded language loop."""

    def __init__(self, child: str, base_path: Optional[Path] = None) -> None:
        self.child = child
        self._base_path = Path(base_path) if base_path else Path("AI_Children")
        self.logger = ExperienceLogger(child=child, base_path=self._base_path)
        self.bridge = LiveExperienceBridge(
            child=child, base_path=self._base_path, logger=self.logger
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_prompt(
        self,
        prompt: str,
        *,
        speaker: str = "operator",
        tags: Optional[Iterable[str]] = None,
        entity_links: Optional[Iterable[Dict[str, Any]]] = None,
        response_tags: Optional[Iterable[str]] = None,
    ) -> str:
        """Process an utterance and craft a grounded reply, logging both turns."""

        base_tags = ["conversation", "lmstudio"]
        inbound_tags = list(base_tags)
        if tags:
            inbound_tags.extend(list(tags))
        inbound_tags = list(dict.fromkeys(inbound_tags))  # preserve order, drop dupes

        entity_payload = list(entity_links) if entity_links else None

        event_id = self.bridge.log_conversation_turn(
            prompt,
            speaker=speaker,
            tags=inbound_tags,
            entity_links=entity_payload,
        )
        response = self._compose_reply(prompt)

        outbound_tags = list(base_tags)
        if response_tags:
            outbound_tags.extend(list(response_tags))
        outbound_tags = list(dict.fromkeys(outbound_tags))

        self.bridge.log_conversation_turn(
            response,
            speaker=self.child,
            tags=outbound_tags,
            entity_links=entity_payload,
            event_id=event_id,
        )
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compose_reply(self, prompt: str) -> str:
        words = self._tokenize(prompt)
        if not words:
            return "I did not catch any words. Could you rephrase that for me?"

        vocab = self._load_known_words()
        grounded_details: List[Tuple[str, Dict[str, Any]]] = []
        unknown_words: List[str] = []
        seen_grounded: set[str] = set()
        seen_unknown: set[str] = set()

        for word in words:
            if word in vocab:
                if word in seen_grounded:
                    continue
                grounding = self._summarise_grounding(word)
                if grounding:
                    grounded_details.append((word, grounding))
                    seen_grounded.add(word)
            elif word not in _STOPWORDS and len(word) > 2 and word not in seen_unknown:
                unknown_words.append(word)
                seen_unknown.add(word)

        if unknown_words:
            for word in unknown_words:
                seed_self_question(
                    f"What experience grounds the word '{word}' mentioned by the operator?"
                )

        if not grounded_details and not unknown_words:
            return (
                "I recognise familiar words, but none are grounded yet. "
                "Could you share an experience that teaches me more?"
            )

        sections: List[str] = []
        if grounded_details:
            sections.append(self._format_grounded_section(grounded_details))

        if unknown_words:
            unique_unknown = unknown_words
            sections.append(
                "I do not have grounding for "
                + ", ".join(f"'{word}'" for word in unique_unknown)
                + ". Could you describe them or show me when they happen?"
            )

        return "\n\n".join(sections)

    def _load_known_words(self) -> Dict[str, str]:
        vocabulary = load_symbol_to_token(self.child, base_path=self._base_path)
        return {
            entry.get("word", "").lower(): symbol
            for symbol, entry in vocabulary.items()
            if entry.get("word")
        }

    def _summarise_grounding(self, word: str) -> Optional[Dict[str, Any]]:
        graph_path = (
            self._base_path
            / self.child
            / "memory"
            / "experiences"
            / "experience_graph.json"
        )
        if not graph_path.exists():
            build_experience_graph(self.child, base_path=self._base_path)

        entries = describe_word_grounding(
            self.child, word, base_path=self._base_path
        )
        if not entries:
            return None

        entry = entries[0]
        narrative = entry.get("narrative") or "I remember the word but not the story."
        narrative = narrative.strip()
        if len(narrative) > 220:
            narrative = narrative[:217].rstrip() + "..."
        return {
            "event_id": entry.get("event_id"),
            "situation_tags": entry.get("situation_tags", []),
            "narrative": narrative,
        }

    def _format_grounded_section(
        self, grounded: Iterable[Tuple[str, Dict[str, Any]]]
    ) -> str:
        lines = ["Here's what I recall:"]
        for word, info in grounded:
            tags = info.get("situation_tags") or []
            tag_text = (
                f" in contexts like {', '.join(sorted(set(tags)))}" if tags else ""
            )
            narrative = info.get("narrative", "")
            lines.append(
                f"- '{word}'{tag_text}. {narrative}".rstrip()
            )
        return "\n".join(lines)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z']+", text)]


class _RequestHandler(BaseHTTPRequestHandler):
    adapter: LMStudioAdapter = None  # type: ignore

    def do_POST(self) -> None:  # noqa: N802 (http method name)
        if self.path != "/v1/chat/completions":
            self._send_response(HTTPStatus.NOT_FOUND, {"error": "Unknown route"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_response(
                HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"}
            )
            return

        prompt = self._extract_prompt(payload)
        if not prompt:
            self._send_response(
                HTTPStatus.BAD_REQUEST, {"error": "Missing user prompt"}
            )
            return

        reply = self.adapter.handle_prompt(prompt)
        response = self._build_completion(payload, reply)
        self._send_response(HTTPStatus.OK, response)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_prompt(self, payload: Dict[str, Any]) -> Optional[str]:
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            return prompt
        return None

    def _build_completion(self, request_payload: Dict[str, Any], reply: str) -> Dict[str, Any]:
        model_name = request_payload.get("model", "ina-symbolic")
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }

    def _send_response(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match signature
        return  # Silence the default noisy logging


def _resolve_child(child_override: Optional[str]) -> str:
    if child_override:
        return child_override
    config = load_config()
    return config.get("current_child", "Inazuma_Yagami")


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> None:
    adapter = LMStudioAdapter(_resolve_child(child), base_path=base_path)
    _RequestHandler.adapter = adapter
    server = ThreadingHTTPServer((host, port), _RequestHandler)
    print(f"[LMAdapter] Serving Ina's chat interface on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LMAdapter] Shutting down adapter.")
    finally:
        server.server_close()


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose Ina through an OpenAI-compatible chat endpoint for LM Studio."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--child",
        help="Override the configured child profile to interact with.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Root path containing AI_Children (defaults to repo storage).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_server(host=args.host, port=args.port, child=args.child, base_path=args.base_path)
