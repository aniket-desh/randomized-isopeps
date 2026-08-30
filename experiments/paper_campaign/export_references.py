"""export only independently validated low-energy reference cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
for source in (repo_root, repo_root / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.paper_campaign.build_manifests import builders, stamp_family_tasks
from rand_isopeps.campaign import finalize_task, runtime_source_fingerprint
from rand_isopeps.campaign.aggregate import read_result_records, validated_references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bond-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    tasks, revision = stamp_family_tasks("references", builders["references"]())
    expected = [finalize_task(task) for task in tasks]
    payload = validated_references(
        read_result_records(args.results),
        bond_tolerance=args.bond_tolerance,
        expected_task_ids=(task["task_id"] for task in expected),
        expected_campaign_revision=revision,
        expected_source_fingerprint=runtime_source_fingerprint(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
