"""Run one explicit bounded reflection/emotion/precision migration batch."""
import argparse
import json
from pathlib import Path

from config_layers import load_config
from witness_event_store import backfill_witness_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("reflection", "emotion", "precision"))
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    cfg = load_config()
    child = str(cfg.get("current_child") or "Inazuma_Yagami")
    memory = Path("AI_Children") / child / "memory"
    filenames = {
        "reflection": "reflection_journal.jsonl",
        "emotion": "emotion_log.jsonl",
        "precision": "precision_memory_map.jsonl",
    }
    result = backfill_witness_batch(
        memory / filenames[args.profile], store=args.profile,
        database=memory / "witness_events.sqlite",
        max_records=args.records, max_bytes=args.bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
