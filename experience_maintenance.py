"""Run one bounded experience-event and live-media condensation cycle."""
import json

from experience_archive import archive_step
from experience_media_archive import media_archive_step
from experience_engine import ExperienceCycleEngine
from storage_layout import load_config


def main() -> int:
    config = load_config()
    child = str(config.get("current_child") or "Inazuma_Yagami")
    result = {
        "events": archive_step(child, config=config),
        "live_media": media_archive_step(child, config=config),
        "experience_cycles": ExperienceCycleEngine(child, config=config).drain_hot_tier(
            max_files=256, max_bytes=16 * 1024 * 1024,
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
