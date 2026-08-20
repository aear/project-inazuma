"""V1/V2 benchmark for capability-scoped resumable file movement."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from storage_migration import managed_migration_step, request_managed_file_move


def main() -> int:
    with TemporaryDirectory(prefix="ina-file-move-benchmark-") as directory:
        root = Path(directory)
        source = root / "AI_Children" / "Ina" / "memory" / "payload.bin"
        target_root = root / "nvme"
        target = target_root / "payload.bin"
        source.parent.mkdir(parents=True)
        payload = b"verified bounded move" * 150_000
        source.write_bytes(payload)
        cfg = {"storage_migration_policy": {"move_target_roots": [str(target_root)]}}
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(root)
            planned = request_managed_file_move("Ina", source, target, choice="inspect", cfg=cfg)
            request_managed_file_move("Ina", source, target, choice="move_and_link", chunk_bytes=1024 * 1024, cfg=cfg)
            state = managed_migration_step("Ina", chunk_bytes=1024 * 1024)
            steps = 1
            while state["status"] in {"copying", "verifying"}:
                state = managed_migration_step("Ina", chunk_bytes=1024 * 1024)
                steps += 1
        finally:
            os.chdir(old_cwd)
        verified = target.read_bytes() == payload and source.is_symlink()
    result = {
        "V1_generic_move": {"choice": 0, "capability_scope": 0, "resume": 0, "verified_cutover": 0},
        "V2_managed_move": {
            "choice": int(planned["status"] == "planned"),
            "capability_scope": int(bool(planned["capabilities"]["target_roots"])),
            "resume": int(steps > 1),
            "verified_cutover": int(state["status"] == "complete" and verified),
            "steps": steps,
        },
    }
    print(result)
    return 0 if all(result["V2_managed_move"][key] == 1 for key in (
        "choice", "capability_scope", "resume", "verified_cutover"
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
