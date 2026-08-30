"""render the paper campaign figures from immutable records."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
for source in (repo_root, repo_root / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.paper_campaign.build_manifests import builders, stamp_family_tasks
from experiments.paper_campaign.plot_columns import (
    plot_candidate_sketches,
    plot_column_oat,
    plot_controlled_spectra,
    plot_isometry_heatmap,
    plot_size_scaling,
)
from experiments.paper_campaign.plot_correctness import plot_correctness
from experiments.paper_campaign.plot_gpu import plot_gpu_crossover
from experiments.paper_campaign.plot_physics import (
    plot_energy_gap,
    plot_physics_sweeps,
    plot_physics_trajectories,
    plot_table_2_comparison,
)
from experiments.paper_campaign.plot_sketches import (
    plot_gaussian_moments_osi,
    plot_rmps_gaussian_ratio,
)
from experiments.paper_campaign.plot_studies import (
    plot_bond_hamiltonian_sweeps,
    plot_dektor_convergence,
    plot_dektor_size_scaling,
)
from experiments.paper_campaign.table_2_validation import is_table_2_exact_problem
from rand_isopeps.campaign import finalize_task, runtime_source_fingerprint
from rand_isopeps.campaign.aggregate import MissingDataError, read_result_records


figures: dict[str, Callable[[Sequence[Mapping], str | Path], Path]] = {
    "gaussian_moments_osi": plot_gaussian_moments_osi,
    "rmps_gaussian_ratio": plot_rmps_gaussian_ratio,
    "size_scaling": plot_size_scaling,
    "candidate_sketches": plot_candidate_sketches,
    "column_oat": plot_column_oat,
    "controlled_spectra": plot_controlled_spectra,
    "isometry_heatmap": plot_isometry_heatmap,
    "correctness": plot_correctness,
    "physics_trajectories": plot_physics_trajectories,
    "energy_gap": plot_energy_gap,
    "physics_sweeps": plot_physics_sweeps,
    "table_2_comparison": plot_table_2_comparison,
    "dektor_convergence": plot_dektor_convergence,
    "dektor_size_scaling": plot_dektor_size_scaling,
    "bond_hamiltonian_sweeps": plot_bond_hamiltonian_sweeps,
    "gpu_crossover": plot_gpu_crossover,
}


def _problem(task: Mapping, key: str, default=None):
    problem = task.get("problem", {})
    return problem.get(key, default) if isinstance(problem, Mapping) else default


def _measurement(task: Mapping, key: str, default=None):
    measurement = task.get("measurement", {})
    return measurement.get(key, default) if isinstance(measurement, Mapping) else default


def _panel(task: Mapping, name: str) -> bool:
    panels = _problem(task, "dektor_panels", ())
    return isinstance(panels, Sequence) and name in panels


figure_requirements = {
    "gaussian_moments_osi": ((
        "gaussian_limit",
        lambda task: _problem(task, "kind") in {"walsh_variance", "walsh_nystrom"},
    ),),
    "rmps_gaussian_ratio": ((
        "gaussian_limit",
        lambda task: _problem(task, "kind") == "sketch_embedding",
    ),),
    "size_scaling": ((
        "column_moves",
        lambda task: _measurement(task, "study") == "method_comparison",
    ),),
    "candidate_sketches": ((
        "column_moves",
        lambda task: _measurement(task, "study") == "method_comparison",
    ),),
    "column_oat": ((
        "column_moves",
        lambda task: _measurement(task, "study") == "one_at_a_time",
    ),),
    "controlled_spectra": ((
        "column_moves",
        lambda task: _measurement(task, "study") == "controlled_spectrum",
    ),),
    "isometry_heatmap": (("isometry", lambda _task: True),),
    "correctness": ((
        "physics",
        lambda task: _problem(task, "study")
        in {"correctness_ladder", "block_correctness"},
    ),),
    "physics_trajectories": ((
        "physics",
        lambda task: _measurement(task, "plot_role") == "physics_trajectory",
    ),),
    "energy_gap": ((
        "physics",
        lambda task: _measurement(task, "plot_role") == "low_energy",
    ),),
    "physics_sweeps": ((
        "physics",
        lambda task: _measurement(task, "plot_role") == "physics_sweep",
    ),),
    "table_2_comparison": (
        ("physics", lambda task: _panel(task, "table_2")),
        (
            "references",
            lambda task: (
                _problem(task, "study") == "exact_reference"
                and is_table_2_exact_problem(task.get("problem", {}))
            ),
        ),
    ),
    "dektor_convergence": ((
        "physics", lambda task: _panel(task, "figure_2")
    ),),
    "dektor_size_scaling": ((
        "physics",
        lambda task: _panel(task, "figure_3") or _panel(task, "figure_4"),
    ),),
    "bond_hamiltonian_sweeps": ((
        "physics",
        lambda task: _problem(task, "study")
        in {"bond_sweep", "hamiltonian_robustness"},
    ),),
    "gpu_crossover": (
        ("gpu_pilot", lambda task: task.get("experiment") == "gpu_pilot"),
        ("gpu_crossover", lambda _task: True),
    ),
}


def validate_current_campaign_coverage(
    rows: Sequence[Mapping], names: Iterable[str]
) -> list[dict]:
    """require complete successful current manifests before final plotting."""
    requested = tuple(names)
    requirements = {
        family: [
            predicate
            for name in requested
            for candidate, predicate in figure_requirements[name]
            if candidate == family
        ]
        for family in {
            candidate
            for name in requested
            for candidate, _predicate in figure_requirements[name]
        }
    }
    source_revision = runtime_source_fingerprint()
    current_ids = set()
    for family, predicates in sorted(requirements.items()):
        tasks, campaign_revision = stamp_family_tasks(family, builders[family]())
        expected = {
            finalize_task(task)["task_id"]
            for task in tasks
            if any(predicate(task) for predicate in predicates)
        }
        current_ids.update(expected)
        selected = [
            row
            for row in rows
            if row.get("campaign_family") == family
            and any(predicate(row) for predicate in predicates)
            and str(row.get("task_id", "")) in expected
        ]
        observed = {str(row.get("task_id", "")) for row in selected}
        missing = expected - observed
        failed = {
            str(row.get("task_id", ""))
            for row in selected
            if row.get("status") != "ok"
        }
        stale = {
            str(row.get("task_id", ""))
            for row in selected
            if row.get("campaign_revision") != campaign_revision
            or row.get("runtime_source_fingerprint") != source_revision
        }
        if missing or failed or stale:
            raise MissingDataError(
                f"incomplete current {family} campaign: missing={len(missing)}, "
                f"failed={len(failed)}, stale={len(stale)}"
            )
    return [dict(row) for row in rows if str(row.get("task_id", "")) in current_ids]


def render_selected(
    rows: Sequence[Mapping], output_dir: str | Path, names: Iterable[str]
) -> list[Path]:
    """render selected figures to paired PDF and PNG outputs."""
    output = Path(output_dir)
    paths = []
    for name in names:
        path = figures[name](rows, output / name)
        print(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=("all", *figures),
        default=[],
    )
    args = parser.parse_args()
    requested = args.figure or ["all"]
    names = tuple(figures) if "all" in requested else tuple(dict.fromkeys(requested))
    rows = read_result_records(args.results)
    current = validate_current_campaign_coverage(rows, names)
    render_selected(current, args.output_dir, names)


if __name__ == "__main__":
    main()
