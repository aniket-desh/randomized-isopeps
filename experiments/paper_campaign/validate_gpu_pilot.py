"""validate cpu/cupy pilot records and write the promotion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
for source in (repo_root, repo_root / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from rand_isopeps.campaign.aggregate import read_result_records
from rand_isopeps.campaign.gpu_gate import validate_gpu_pilot
from rand_isopeps.campaign.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    payload = validate_gpu_pilot(
        read_result_records(args.results),
        tolerance=args.tolerance,
        expected_tasks=read_manifest(args.manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
