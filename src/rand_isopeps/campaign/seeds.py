"""stable paired seed streams for problem, sketch, score, and timing draws."""

from __future__ import annotations

import hashlib
import json


def derive_seed(root_seed: int, *labels) -> int:
    """derive a platform-independent nonnegative 63-bit seed."""
    root = int(root_seed)
    if root < 0:
        raise ValueError("root_seed must be nonnegative")
    payload = json.dumps(
        [root, *labels],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def seed_hierarchy(
    root_seed: int,
    problem_index: int,
    sketch_index: int = 0,
    score_index: int = 0,
    timing_index: int = 0,
) -> dict[str, int]:
    """build independent streams nested under one paired problem seed."""
    problem = derive_seed(root_seed, "problem", int(problem_index))
    return {
        "root": int(root_seed),
        "problem": problem,
        "sketch": derive_seed(problem, "sketch", int(sketch_index)),
        "score": derive_seed(problem, "score", int(score_index)),
        "timing": derive_seed(problem, "timing", int(timing_index)),
    }


def seed_stream(root_seed: int, label: str, count: int) -> tuple[int, ...]:
    """return a deterministic sequence without mutable generator state."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    return tuple(derive_seed(root_seed, label, index) for index in range(int(count)))
