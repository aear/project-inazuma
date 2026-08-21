"""V1/V2 benchmark for optional and verified evidence maintenance."""
from __future__ import annotations

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

from storage_maintenance import maintenance_opportunity, perform_choice


def main() -> int:
    with TemporaryDirectory(prefix="ina-maintenance-benchmark-") as directory:
        root = Path(directory)
        logs = root / "logs"
        logs.mkdir()
        source = logs / "ina_status.log.1"
        payload = b"long-lived operational evidence\n" * 600_000
        source.write_bytes(payload)
        opportunity = maintenance_opportunity(root)
        deferred = perform_choice(root, "defer")
        retained_after_defer = source.exists()
        compressed = perform_choice(root, "compress_one")
        target = source.with_name(source.name + ".gz")
        verified_payload = gzip.open(target, "rb").read() == payload
    result = {
        "V1_manual_unverified": {"choice": 0, "hash_verified": 0, "bounded_files": 0},
        "V2_optional_verified": {
            "choice": int(opportunity["available"] and deferred["status"] == "deferred" and retained_after_defer),
            "hash_verified": int(compressed.get("verified") is True and verified_payload),
            "bounded_files": int(opportunity["candidate_count"] == 1),
        },
    }
    print(result)
    return 0 if all(result["V2_optional_verified"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
