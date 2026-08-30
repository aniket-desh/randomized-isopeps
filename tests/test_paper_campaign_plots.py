import pytest

from paper_campaign_plot_fixtures import extended_figure_rows
from experiments.paper_campaign.build_manifests import builders, stamp_family_tasks
from experiments.paper_campaign.plot_results import (
    figures,
    render_selected,
    validate_current_campaign_coverage,
)
from experiments.paper_campaign.plot_common import final_converged_measurements
from experiments.paper_campaign.plot_physics import _table_2_executed_bands
from experiments.paper_campaign.plot_sketches import _pooled_variance
from experiments.paper_campaign.table_2_validation import (
    table_2_references,
    validate_table_2_exact_references,
)
from rand_isopeps.campaign.aggregate import MissingDataError
from rand_isopeps.campaign import finalize_task


def _record(experiment, task_id, seed, **values):
    return {
        "experiment": experiment,
        "task_id": task_id,
        "status": "ok",
        "backend": values.pop("backend", "numpy"),
        "seeds": {"problem": seed},
        **values,
    }


def _gaussian_rows():
    rows = []
    for lx in (2, 3):
        for seed in (0, 1):
            problem_seed = 100 * lx + seed
            common = {
                "benchmark": "column_embedding",
                "lx": lx,
                "subspace": "haar",
                "ell": 4,
                "rank": 2,
                "osi_sigma_min": 0.9,
            }
            rows.append(
                _record(
                    "gaussian_limit",
                    f"g-{lx}-{seed}",
                    problem_seed,
                    method="global_gaussian",
                    chi_sk=0,
                    projection_error=0.1,
                    normalized_fourth_moment_error=0.02,
                    **common,
                )
            )
            for chi_sk, error in ((2, 0.2), (4, 0.12)):
                rows.append(
                    _record(
                        "gaussian_limit",
                        f"r-{lx}-{seed}-{chi_sk}",
                        problem_seed,
                        method="rmps",
                        chi_sk=chi_sk,
                        projection_error=error,
                        normalized_fourth_moment_error=error,
                        **{**common, "osi_sigma_min": 0.5 + 0.05 * chi_sk},
                    )
                )
    for trial in range(3):
        rows.append(
            _record(
                "gaussian_limit",
                f"v-g-{trial}",
                trial,
                benchmark="rmps_figure2_variance",
                method="global_gaussian",
                chi_sk=0,
                normalized_quadratic_variance=0.1 + trial * 1e-3,
            )
        )
        for chi_sk in (1, 10, 20):
            rows.append(
                _record(
                    "gaussian_limit",
                    f"v-r-{chi_sk}-{trial}",
                    trial,
                    benchmark="rmps_figure2_variance",
                    method="rmps",
                    chi_sk=chi_sk,
                    normalized_quadratic_variance=0.1 + 1.0 / chi_sk,
                )
            )
        for embedding_dim in (4, 8, 16):
            rows.append(
                _record(
                    "gaussian_limit",
                    f"n-g-{embedding_dim}-{trial}",
                    trial,
                    benchmark="rmps_figure2_nystrom",
                    method="gaussian_nystrom",
                    chi_sk=0,
                    embedding_dim=embedding_dim,
                    relative_nuclear_error=1.0 / embedding_dim,
                )
            )
            for chi_sk in (1, 10):
                rows.append(
                    _record(
                        "gaussian_limit",
                        f"n-r-{chi_sk}-{embedding_dim}-{trial}",
                        trial,
                        benchmark="rmps_figure2_nystrom",
                        method="mps_gram_nystrom",
                        chi_sk=chi_sk,
                        embedding_dim=embedding_dim,
                        relative_nuclear_error=(1.0 + 1.0 / chi_sk) / embedding_dim,
                    )
                )
    return rows


def _column_rows():
    methods = (
        ("sequential_moses", "local_det_ndis0"),
        ("sequential_moses_riemannian", "local_riemannian_ndis30"),
        ("local_gaussian", "local_rsvd2_gaussian"),
        ("local_sparsestack", "local_rsvd2_sparsestack"),
        ("global_gaussian", "global_gaussian"),
        ("global_rademacher", "global_rademacher"),
        ("global_sparsestack", "global_sparsestack"),
        ("global_rmps", "global_rmps_bounded"),
        ("global_kron", "global_kron"),
    )
    rows = []
    for lx in (2, 3):
        for state in ("random", "tfim@3.5"):
            for seed in (0, 1):
                problem_seed = 1_000 * lx + 10 * seed + (state != "random")
                for index, (method, label) in enumerate(methods):
                    rows.append(
                        _record(
                            "column_moves",
                            f"c-{lx}-{state}-{seed}",
                            problem_seed,
                            method=method,
                            method_label=label,
                            state=state,
                            lx=lx,
                            state_infidelity=(index + 1) * lx * 1e-6,
                            projection_error=(index + 1) * 1e-4,
                        )
                    )
    return rows


def _isometry_rows():
    methods = (
        "local_det_ndis0",
        "local_riemannian_ndis30",
        "local_rsvd2_gaussian",
        "local_rsvd2_sparsestack",
        "global_gaussian",
        "global_rademacher",
        "global_sparsestack",
        "global_rmps_bounded",
        "global_kron",
    )
    states = ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
    routes = (
        "boundary_right",
        "interior_right",
        "interior_left",
        "round_trip",
    )
    rows = []
    for state_index, state in enumerate(states):
        for method_index, method in enumerate(methods):
            for route_index, route in enumerate(routes):
                for seed in (0, 1, 2):
                    rows.append(
                        _record(
                            "isometry",
                            f"i-{state}-{method}-{route}-{seed}",
                            seed,
                            lx=2,
                            ly=3,
                            state=state,
                            regime="truncated",
                            method_label=method,
                            route=route,
                            max_local_isometry_defect=(
                                state_index
                                + method_index
                                + route_index
                                + seed
                                + 1
                            )
                            * 1e-12,
                        )
                    )
    return rows


def _oat_rows():
    baseline = {"eta": 4, "ell": 8, "chi_sk": 8, "kappa": 2, "n_power": 0}
    variants = {
        "eta": 8,
        "ell": 12,
        "chi_sk": 16,
        "kappa": 4,
        "n_power": 1,
    }
    rows = []
    for lx in (4, 6, 7):
        for state_index, state in enumerate(
            ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
        ):
            configs = [baseline]
            configs.extend({**baseline, axis: value} for axis, value in variants.items())
            for config_index, config in enumerate(configs):
                for replicate in (0, 1):
                    rows.append(
                        _record(
                            "column_moves",
                            f"oat-{lx}-{state}-{config_index}-{replicate}",
                            100_000 * lx + 1_000 * state_index + replicate,
                            study="one_at_a_time",
                            state=state,
                            lx=lx,
                            baseline=baseline,
                            method_config=config,
                            state_infidelity=(config_index + state_index + 1)
                            * 1e-6,
                        )
                    )
    return rows


def _physics_rows():
    rows = []
    for lx, chi in ((2, 4), (3, 8)):
        for seed in (0, 1):
            problem_seed = 10_000 * lx + seed
            for method_index, method in enumerate(("dense_strang", "peps_sketch")):
                for iteration in (0, 1, 2):
                    error = (method_index + 1) * 1e-2 / (iteration + 1)
                    rows.append(
                        _record(
                            "physics",
                            f"p-{lx}-{seed}-{method}",
                            problem_seed,
                            method=method,
                            method_label=method,
                            hamiltonian="tfim@3.5",
                            study="synthetic_trajectory",
                            lx=lx,
                            ly=lx,
                            chi=chi,
                            states=2,
                            iteration=iteration,
                            energies=[-1.0 - 0.1 * lx, -0.7 - 0.1 * lx],
                            reference_energies=[
                                -1.01 - 0.1 * lx,
                                -0.72 - 0.1 * lx,
                            ],
                            ground_energy_errors=[error, error * 2],
                            relative_residuals=[error * 0.1, error * 0.2],
                        )
                    )
    return rows


def _correctness_rows():
    rows = []
    single_methods = (
        "dense_exact",
        "dense_strang",
        "peps_full",
        "peps_local",
        "peps_sketch",
    )
    for hamiltonian in ("tfim@3.5", "heis"):
        for lx, ly in ((2, 2), (2, 3), (3, 3)):
            for tau in (0.1, 0.05, 0.025, 0.0125):
                seed = hash((hamiltonian, lx, ly, tau)) % 10_000
                exact = -float(lx * ly) - tau
                for method_index, method in enumerate(single_methods):
                    rows.append(_record(
                        "physics",
                        f"single-{hamiltonian}-{lx}-{ly}-{tau}-{method}",
                        seed,
                        method=method,
                        study="correctness_ladder",
                        plot_role="correctness",
                        hamiltonian=hamiltonian,
                        lx=lx,
                        ly=ly,
                        states=1,
                        iteration=1,
                        tau=tau,
                        energies=[exact + method_index * tau * 1e-3],
                    ))
    block_configs = (
        ("dense_exact", "dense_oracle"),
        ("dense_first_order", "dense_oracle"),
        ("peps_local", "full_rank_oracle"),
        ("peps_sketch", "full_rank_oracle"),
        ("peps_local", "truncated"),
        ("peps_sketch", "truncated"),
    )
    for hamiltonian in ("tfim@3.5", "heis"):
        for lx, ly, states in ((2, 2, 2), (2, 3, 2), (2, 2, 3)):
            seed = hash((hamiltonian, lx, ly, states)) % 10_000
            exact = [-float(lx * ly) + 0.2 * index for index in range(states)]
            for method_index, (method, regime) in enumerate(block_configs):
                rows.append(_record(
                    "physics",
                    f"block-{hamiltonian}-{lx}-{ly}-{states}-{method}-{regime}",
                    seed,
                    method=method,
                    study="block_correctness",
                    plot_role="correctness",
                    hamiltonian=hamiltonian,
                    lx=lx,
                    ly=ly,
                    states=states,
                    regime=regime,
                    iteration=4,
                    energies=[
                        value + method_index * (state_index + 1) * 1e-3
                        for state_index, value in enumerate(exact)
                    ],
                ))
    return rows


def _gpu_rows():
    rows = []
    for lx in (5, 6):
        for seed in (0, 1):
            problem_seed = 100_000 * lx + seed
            for backend, runtime in (("numpy", 0.2 * lx), ("cupy", 0.05 * lx)):
                rows.append(
                    _record(
                        "gpu_pilot",
                        f"gpu-{lx}-{seed}-{backend}",
                        problem_seed,
                        backend=backend,
                        lx=lx,
                        mpo_bond=4,
                        ell=20,
                        chi_sk=8,
                        n_power=0,
                        replicate=seed,
                        matrix_products=8,
                        median_runtime_s=runtime,
                    )
                )
    return rows


def test_core_figures_render_from_synthetic_records(tmp_path):
    rows = (
        _gaussian_rows()
        + _column_rows()
        + _isometry_rows()
        + _physics_rows()
        + _correctness_rows()
        + _gpu_rows()
    )
    rows.extend((
        _record(
            "column_moves",
            "unrelated-column",
            999,
            study="one_at_a_time",
            measurement_converged=False,
            campaign_family="column_moves",
            campaign_revision="old-column",
            runtime_source_fingerprint="old-source",
        ),
        _record(
            "physics",
            "unrelated-physics",
            999,
            plot_role="unused",
            measurement_converged=False,
            campaign_family="physics",
            campaign_revision="old-physics",
            runtime_source_fingerprint="old-source",
        ),
    ))
    core = (
        "gaussian_moments_osi",
        "rmps_gaussian_ratio",
        "size_scaling",
        "candidate_sketches",
        "isometry_heatmap",
        "correctness",
        "physics_trajectories",
        "energy_gap",
        "gpu_crossover",
    )
    paths = render_selected(rows, tmp_path, core)
    assert len(paths) == len(core)
    for name in core:
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_every_registered_figure_renders_from_complete_synthetic_records(tmp_path):
    rows = (
        _gaussian_rows()
        + _column_rows()
        + _isometry_rows()
        + _oat_rows()
        + _physics_rows()
        + _correctness_rows()
        + _gpu_rows()
        + extended_figure_rows()
    )

    paths = render_selected(rows, tmp_path, figures)

    assert {path.stem for path in paths} == set(figures)
    for name in figures:
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_plot_registry_covers_every_expensive_study():
    assert set(figures) == {
        "gaussian_moments_osi",
        "rmps_gaussian_ratio",
        "size_scaling",
        "candidate_sketches",
        "column_oat",
        "controlled_spectra",
        "isometry_heatmap",
        "correctness",
        "physics_trajectories",
        "energy_gap",
        "physics_sweeps",
        "table_2_comparison",
        "dektor_convergence",
        "dektor_size_scaling",
        "bond_hamiltonian_sweeps",
        "gpu_crossover",
    }


def test_column_oat_handles_disjoint_state_seeds(tmp_path):
    path = figures["column_oat"](_oat_rows(), tmp_path / "column-oat")
    assert path.stat().st_size > 0


def test_plot_missing_data_is_explicit(tmp_path):
    with pytest.raises(MissingDataError, match="rmps figure 2"):
        figures["gaussian_moments_osi"]([], tmp_path / "missing")


def test_accuracy_plot_rejects_unconverged_preparation(tmp_path):
    rows = _column_rows()
    rows[0]["preparation"] = {"converged": False}
    with pytest.raises(MissingDataError, match="unconverged measurement"):
        figures["size_scaling"](rows, tmp_path / "bad-preparation")


def test_final_measurement_gate_ignores_only_unconverged_intermediate_rows():
    rows = [
        _record(
            "physics",
            "trajectory",
            1,
            iteration=0,
            measurement_converged=False,
        ),
        _record(
            "physics",
            "trajectory",
            1,
            iteration=1,
            measurement_converged=True,
        ),
    ]

    assert final_converged_measurements(rows)[0]["iteration"] == 1
    rows[-1]["measurement_converged"] = False
    with pytest.raises(MissingDataError, match="unconverged measurement"):
        final_converged_measurements(rows)


def test_final_plot_coverage_rejects_partial_current_family():
    with pytest.raises(MissingDataError, match="incomplete current gaussian_limit"):
        validate_current_campaign_coverage([], ("gaussian_moments_osi",))


def test_correctness_coverage_requires_every_single_and_block_task():
    tasks, _ = stamp_family_tasks("physics", builders["physics"]())
    expected = [
        finalize_task(task)
        for task in tasks
        if task["problem"]["study"] in {"correctness_ladder", "block_correctness"}
    ]
    rows = [{**task, "status": "ok"} for task in expected]

    current = validate_current_campaign_coverage(rows, ("correctness",))

    assert len(expected) == 276
    assert {row["task_id"] for row in current} == {
        task["task_id"] for task in expected
    }
    with pytest.raises(MissingDataError, match="incomplete current physics"):
        validate_current_campaign_coverage(rows[:-1], ("correctness",))


def test_final_plot_coverage_is_scoped_to_the_requested_figure():
    tasks, _ = stamp_family_tasks("gaussian_limit", builders["gaussian_limit"]())
    embedding = [
        {**finalize_task(task), "status": "ok"}
        for task in tasks
        if task["problem"]["kind"] == "sketch_embedding"
    ]
    unrelated = next(
        {**finalize_task(task), "status": "failed"}
        for task in tasks
        if task["problem"]["kind"] == "walsh_variance"
    )

    stale = {
        **embedding[0],
        "task_id": "old-embedding-task",
        "status": "failed",
        "campaign_revision": "old-campaign",
        "runtime_source_fingerprint": "old-source",
    }

    current = validate_current_campaign_coverage(
        [*embedding, unrelated, stale],
        ("rmps_gaussian_ratio",),
    )

    assert {row["task_id"] for row in current} == {
        row["task_id"] for row in embedding
    }


def test_table_2_executed_curves_pool_duplicate_replicates():
    rows = [
        {
            "status": "ok",
            "state_index": 0,
            "method": "global_rmps_bounded",
            "chi": 12,
            "eta": 20,
            "g": g,
            "value": value,
        }
        for g, value in (
            (3.0, 0.3),
            (3.0, 0.1),
            (3.0, 0.2),
            (3.5, 0.06),
            (3.5, 0.04),
            (3.5, 0.05),
        )
    ]
    band = _table_2_executed_bands(rows, 0)[
        ("global_rmps_bounded", 12, 20)
    ]

    assert band["x"] == [3.0, 3.5]
    assert band["median"] == pytest.approx([0.2, 0.05])
    assert band["n"] == [3, 3]


def test_table_2_published_energies_require_complete_exact_cross_check():
    rows = [
        {
            "hamiltonian": hamiltonian,
            "lx": 4,
            "ly": 4,
            "states": 2,
            "energies": energies,
            "reference_tier": "exact",
            "validation_passed": True,
        }
        for hamiltonian, energies in table_2_references().items()
    ]
    next(row for row in rows if row["hamiltonian"] == "tfim@1")["energies"] = [
        -26.860504639527907,
        -26.860427061085538,
    ]
    next(row for row in rows if row["hamiltonian"] == "tfim@3.5")[
        "energies"
    ].append(-50.0)
    evidence = validate_table_2_exact_references(rows)
    assert evidence["states"] == 8
    assert evidence["max_relative_difference"] < 2e-6

    rows.pop()
    with pytest.raises(ValueError, match="coverage is missing"):
        validate_table_2_exact_references(rows)


def test_rmps_variance_centerline_pools_all_trial_samples():
    trials = [
        {
            "samples": 2,
            "quadratic_sample_mean": mean,
            "quadratic_sample_m2": 2.0,
            "trace_value": 1.0,
        }
        for mean in (0.0, 2.0)
    ]
    assert _pooled_variance(trials) == pytest.approx(8.0 / 3.0)
