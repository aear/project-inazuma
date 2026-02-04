import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experience_logger import ExperienceLogger
from memory_graph import build_experience_graph
from model_manager import load_config


def _load_default_child() -> str:
    try:
        cfg = load_config()
    except Exception:
        return "Inazuma_Yagami"
    return cfg.get("current_child", "Inazuma_Yagami")


def _load_deck(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Deck not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {}, [card for card in data if isinstance(card, dict)]
    if isinstance(data, dict):
        cards = data.get("cards", [])
        if isinstance(cards, list):
            return data, [card for card in cards if isinstance(card, dict)]
    return {}, []


def _normalize_words(card: Dict[str, Any]) -> List[str]:
    words = card.get("words")
    if isinstance(words, list):
        clean = [str(w).strip() for w in words if str(w).strip()]
        if clean:
            return clean
    word = str(card.get("word", "")).strip()
    return [word] if word else []


def _build_entities(card: Dict[str, Any], words: List[str]) -> List[Dict[str, Any]]:
    entities = card.get("entities")
    if isinstance(entities, list):
        clean = [e for e in entities if isinstance(e, dict)]
        if clean:
            return clean
    if words:
        return [{"id": f"card:{words[0]}", "label": words[0], "type": "flashcard"}]
    return []


def _build_media_entities(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    media_entities = []
    image_path = card.get("image")
    if image_path:
        media_entities.append({"type": "image", "path": str(image_path), "label": "flashcard_image"})
    audio_path = card.get("audio")
    if audio_path:
        media_entities.append({"type": "audio", "path": str(audio_path), "label": "flashcard_audio"})
    return media_entities


def _log_card(
    logger: ExperienceLogger,
    card: Dict[str, Any],
    *,
    deck_id: Optional[str] = None,
    speaker: str = "operator",
    grade: bool = False,
) -> Optional[str]:
    words = _normalize_words(card)
    if not words:
        return None

    tags = ["flashcard", "language", "study"]
    card_tags = card.get("tags")
    if isinstance(card_tags, list):
        tags.extend(str(tag) for tag in card_tags if str(tag))

    entities = _build_entities(card, words)
    entities.extend(_build_media_entities(card))

    prompt = str(card.get("prompt") or card.get("utterance") or words[0]).strip()
    utterance = str(card.get("utterance") or words[0]).strip()
    card_id = str(card.get("id") or f"card_{words[0]}").strip()

    internal_state = {
        "flashcard": {
            "deck_id": deck_id,
            "card_id": card_id,
            "prompt": prompt,
            "word": words[0],
        }
    }

    actions = [{"verb": "review", "object": card_id}]
    outcome: Dict[str, Any] = {}
    if grade:
        response = input("Mark correct? [y/N]: ").strip().lower()
        outcome["correct"] = response in {"y", "yes"}

    event_id = logger.log_event(
        situation_tags=tags,
        perceived_entities=entities,
        actions=actions,
        outcome=outcome,
        internal_state=internal_state,
        narrative=f"Flashcard review: {prompt}",
    )

    logger.attach_word_usage(
        event_id,
        speaker=speaker,
        utterance=utterance,
        words=words,
        entity_links=[{"entity_id": ent.get("id")} for ent in entities if ent.get("id")],
    )
    return event_id


def run_session(
    cards: List[Dict[str, Any]],
    *,
    child: str,
    base_path: Optional[Path],
    deck_id: Optional[str],
    speaker: str,
    interactive: bool,
    shuffle: bool,
    limit: Optional[int],
    delay: float,
    grade: bool,
    build_graph: bool,
) -> int:
    if shuffle:
        random.shuffle(cards)
    if limit is not None:
        cards = cards[:limit]

    logger = ExperienceLogger(child=child, base_path=base_path)
    logged = 0

    for idx, card in enumerate(cards, start=1):
        words = _normalize_words(card)
        label = words[0] if words else "untitled"
        prompt = str(card.get("prompt") or card.get("utterance") or label).strip()

        print(f"[Flashcards] {idx}/{len(cards)} → {label}")
        print(f"[Flashcards] Prompt: {prompt}")

        if interactive:
            response = input("Enter to log, 's' to skip, 'q' to quit: ").strip().lower()
            if response in {"q", "quit"}:
                break
            if response in {"s", "skip"}:
                continue

        event_id = _log_card(
            logger,
            card,
            deck_id=deck_id,
            speaker=speaker,
            grade=grade,
        )
        if event_id:
            logged += 1

        if delay > 0:
            time.sleep(delay)

    if build_graph:
        build_experience_graph(child, base_path=base_path)

    return logged


def main() -> None:
    parser = argparse.ArgumentParser(description="Flashcard-driven experience grounding.")
    parser.add_argument("--deck", default="flashcards/ina_flashcards.json")
    parser.add_argument("--child", default=_load_default_child())
    parser.add_argument("--base-path", default=None)
    parser.add_argument("--speaker", default="operator")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()

    deck_path = Path(args.deck)
    base_path = Path(args.base_path) if args.base_path else None
    meta, cards = _load_deck(deck_path)
    deck_id = meta.get("deck_id") if isinstance(meta, dict) else None

    if not cards:
        raise SystemExit(f"No cards found in deck: {deck_path}")

    logged = run_session(
        cards,
        child=args.child,
        base_path=base_path,
        deck_id=deck_id,
        speaker=args.speaker,
        interactive=not args.auto,
        shuffle=args.shuffle,
        limit=args.limit,
        delay=max(0.0, args.delay),
        grade=args.grade,
        build_graph=not args.no_graph,
    )

    print(f"[Flashcards] Logged {logged} card(s).")


if __name__ == "__main__":
    main()
