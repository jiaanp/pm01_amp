#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lock = json.loads((ROOT / "provenance" / "source_lock.json").read_text())
    assert lock["amp_mjlab"]["commit"] == "6c7a2947fccc973e4af8e6d90e550400f1b6fcfc"
    assert lock["engineai_amp"]["commit"] == "83ba64bbb58a02e14483e52adce5f893f3f31cdf"
    assert lock["gmr"]["commit"] == "bb1bbe40774794fceb2a7c579a3464a28e68c844"

    entries = []
    for line in (ROOT / "provenance" / "source_dataset.sha256").read_text().splitlines():
        if line.strip():
            expected, relative = line.split(maxsplit=1)
            path = ROOT / relative
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"SHA256 mismatch: {relative}: {actual} != {expected}")
            entries.append(relative)

    urdf = (ROOT / "assets" / "pm01" / "urdf" / "serial_pm01.urdf").read_text()
    if '<joint name="J23_HEAD_YAW" type="revolute">' not in urdf:
        raise RuntimeError("The approved 24-DoF head adaptation is missing.")
    print(f"Provenance OK: {len(entries)} source motions, locked commits, 24-DoF URDF.")


if __name__ == "__main__":
    main()

