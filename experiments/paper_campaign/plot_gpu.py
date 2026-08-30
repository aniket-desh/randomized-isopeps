"""optional CPU/GPU crossover figure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from rand_isopeps.campaign.aggregate import (
    field,
    finite_float,
    median_bands,
    paired_median_bands,
    require_rows,
)
from rand_isopeps.plotting import apply_paper_style, clean_axis

from .plot_common import draw_bands, finish, need_bands, validate_plot_revisions

import matplotlib.pyplot as plt


def plot_gpu_crossover(rows: Sequence[Mapping], path: str | Path) -> Path:
    data = require_rows(
        rows,
        lambda row: row.get("experiment") in {"gpu_pilot", "gpu_crossover"}
        and row.get("status") == "ok"
        and finite_float(row.get("median_runtime_s")) is not None,
        "cpu-gpu crossover",
    )
    data = validate_plot_revisions(data, "cpu-gpu crossover")
    paired = defaultdict(dict)
    for row in data:
        runtime = finite_float(row.get("median_runtime_s"))
        lx = finite_float(row.get("lx"))
        mpo_bond = finite_float(row.get("mpo_bond"))
        seed = field(row, "seeds.problem", None)
        backend = str(row.get("backend"))
        ell = finite_float(row.get("ell"))
        chi_sk = finite_float(row.get("chi_sk"))
        n_power = finite_float(row.get("n_power"))
        replicate = row.get("replicate")
        replicate_seed = field(row, "replicate_seeds.problem", None)
        if None in (runtime, lx, mpo_bond, ell, chi_sk, n_power, seed):
            continue
        key = (
            seed,
            replicate,
            replicate_seed,
            lx,
            mpo_bond,
            ell,
            chi_sk,
            n_power,
        )
        paired[key][backend] = runtime
    runtime_rows = []
    for pair_index, values in enumerate(paired.values()):
        if "numpy" not in values:
            continue
        runtime_rows.extend(
            {
                "status": "ok",
                "backend": backend,
                "cpu_runtime": values["numpy"],
                "runtime": runtime,
                "seeds": {"problem": pair_index},
            }
            for backend, runtime in sorted(values.items())
        )
    runtime_bands = need_bands(
        paired_median_bands(
            runtime_rows,
            "backend",
            "cpu_runtime",
            "runtime",
            groups=("numpy", "cupy"),
        ),
        "paired cpu and gpu runtimes",
    )
    speedups = [
        {
            "status": "ok",
            "group": "cpu / gpu",
            "cpu_runtime": values["numpy"],
            "speedup": values["numpy"] / values["cupy"],
        }
        for values in paired.values()
        if "numpy" in values and "cupy" in values and values["cupy"] > 0
    ]
    speedup_bands = need_bands(
        median_bands(speedups, "group", "cpu_runtime", "speedup"),
        "paired cpu-gpu speedups",
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    draw_bands(
        axes[0],
        runtime_bands,
        styles=lambda group, _index: (
            ("cpu", "#0072b2", "o", "-")
            if group == "numpy"
            else ("gpu", "#d55e00", "s", "-")
        ),
    )
    axes[0].set_xlabel("paired CPU runtime (s)")
    axes[0].set_ylabel("synchronized runtime (s)")
    clean_axis(axes[0], ylog=True, xlog=True)
    axes[0].legend()
    draw_bands(
        axes[1],
        speedup_bands,
        styles=lambda _group, _index: ("cpu / gpu", "#009e73", "D", "-"),
    )
    axes[1].axhline(1.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("paired CPU runtime (s)")
    axes[1].set_ylabel("speedup")
    clean_axis(axes[1], xlog=True)
    return finish(fig, path)
