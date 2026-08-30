"""independent exact checks for the published dektor table 2 energies."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

table_2_exact_relative_tolerance = 2e-6
table_2_path = Path(__file__).parent / "references" / "dektor_table_2_v1.json"


def table_2_references() -> dict[str, list[float]]:
    """load published table 2 references from the versioned literature artifact."""
    payload = json.loads(table_2_path.read_text(encoding="utf-8"))
    grouped = {}
    for row in payload["records"]:
        key = f"tfim@{float(row['g']):g}"
        grouped.setdefault(key, []).append(
            (int(row["state_index"]), float(row["reference_energy"]))
        )
    return {
        key: [energy for _, energy in sorted(values)]
        for key, values in grouped.items()
    }


def is_table_2_exact_problem(problem: Mapping) -> bool:
    """identify a 4x4 exact-reference cell used by dektor table 2."""
    return (
        int(problem.get("lx", -1)) == 4
        and int(problem.get("ly", -1)) == 4
        and str(problem.get("hamiltonian", "")) in table_2_references()
    )


def validate_table_2_exact_references(
    records: Iterable[Mapping],
    *,
    relative_tolerance: float = table_2_exact_relative_tolerance,
) -> dict:
    """require every published table 2 energy to agree with exact diagonalization."""
    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("table 2 exact-reference tolerance must be positive and finite")

    exact = {}
    for row in records:
        if (
            row.get("validation_passed") is not True
            or str(row.get("reference_tier", "")) != "exact"
            or not is_table_2_exact_problem(row)
        ):
            continue
        energies = row.get("energies")
        if not isinstance(energies, Sequence) or isinstance(energies, (str, bytes)):
            continue
        try:
            values = [float(value) for value in energies]
        except (TypeError, ValueError):
            continue
        if len(values) < 2 or not all(math.isfinite(value) for value in values[:2]):
            continue
        hamiltonian = str(row["hamiltonian"])
        if hamiltonian not in exact or len(values) > len(exact[hamiltonian]):
            exact[hamiltonian] = values

    published = table_2_references()
    missing = sorted(set(published) - set(exact))
    if missing:
        raise ValueError(
            "table 2 exact-reference coverage is missing: " + ", ".join(missing)
        )

    differences = {}
    for hamiltonian, expected in sorted(published.items()):
        exact_values = exact[hamiltonian][: len(expected)]
        for state_index, (published_energy, exact_energy) in enumerate(
            zip(expected, exact_values, strict=True)
        ):
            scale = max(abs(published_energy), abs(exact_energy), 1.0)
            difference = abs(published_energy - exact_energy) / scale
            differences[(hamiltonian, state_index)] = difference
            if difference > tolerance:
                raise ValueError(
                    "published table 2 energy disagrees with exact diagonalization: "
                    f"{hamiltonian}, state={state_index}, relative_difference="
                    f"{difference:.3e}, tolerance={tolerance:.3e}"
                )
    return {
        "cells": len(published),
        "states": len(differences),
        "relative_tolerance": tolerance,
        "max_relative_difference": max(differences.values(), default=0.0),
    }
