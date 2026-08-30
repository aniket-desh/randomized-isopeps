"""synthetic records for complete paper-campaign plot qa."""

from experiments.paper_campaign.plot_columns import method_order
from experiments.paper_campaign.plot_studies import comparison_methods
from experiments.paper_campaign.table_2_validation import (
    table_2_exact_relative_tolerance,
    table_2_references,
)


def _record(experiment, task_id, seed, **values):
    return {
        "experiment": experiment,
        "task_id": task_id,
        "status": "ok",
        "backend": values.pop("backend", "numpy"),
        "seeds": {"problem": seed},
        **values,
    }


def _energy_values(states, error, references=None):
    exact = references or [-10.0 + 0.25 * index for index in range(states)]
    errors = [error * (index + 1) for index in range(states)]
    return {
        "energies": [value + delta for value, delta in zip(exact, errors)],
        "reference_energies": list(exact),
        "ground_energy_errors": errors,
    }


def _physics_record(task_id, seed, states, error, **values):
    references = values.pop("reference_energies", None)
    return _record(
        "physics",
        task_id,
        seed,
        states=states,
        measurement_converged=True,
        **_energy_values(states, error, references),
        **values,
    )


def controlled_spectrum_rows():
    rows = []
    controls = (("controlled_exp", 4.0), ("controlled_power", 1.0))
    methods = tuple(name for name in method_order if name.startswith("global_"))
    for family_index, (family, decay) in enumerate(controls):
        for lx in (2, 4):
            for replicate in (0, 1):
                seed = 1000 * family_index + 10 * lx + replicate
                for method_index, method in enumerate(methods):
                    rows.append(_record(
                        "column_moves",
                        f"spectrum-{family}-{lx}-{replicate}-{method}",
                        seed,
                        study="controlled_spectrum",
                        family=family,
                        decay=decay,
                        lx=lx,
                        method_label=method,
                        projection_excess=(method_index + 1) * lx * 1e-6,
                    ))
    return rows


def _sketch_parameter_rows():
    rows = []
    for axis, values in (("ell", (8, 16)), ("chi_sk", (4, 8)), ("kappa", (1, 2))):
        for x in values:
            for states in (1, 2):
                for replicate in (0, 1):
                    seed = 10_000 + 100 * int(x) + 10 * states + replicate
                    rows.append(_physics_record(
                        f"sketch-{axis}-{x}-{states}-{replicate}",
                        seed,
                        states,
                        1e-3 / (float(x) + states),
                        study="sketch_parameter_sweep",
                        plot_role="physics_sweep",
                        sweep_axis=axis,
                        method="peps_sketch",
                        method_label="global_rmps_bounded",
                        method_config={axis: x},
                        iteration=4,
                    ))
    return rows


def _bond_parameter_rows():
    rows = []
    methods = (("peps_local", "local_det"), ("peps_sketch", "global_rmps_bounded"))
    for axis, values in (("chi", (4, 8)), ("eta", (8, 16))):
        for x in values:
            for replicate in (0, 1):
                seed = 20_000 + 100 * int(x) + replicate
                for method_index, (method, label) in enumerate(methods):
                    chi, eta = (x, 2 * x) if axis == "chi" else (4, x)
                    rows.append(_physics_record(
                        f"bond-parameter-{axis}-{x}-{replicate}-{method}",
                        seed,
                        1,
                        (method_index + 1) * 1e-3 / float(x),
                        study="bond_parameter_sweep",
                        plot_role="physics_sweep",
                        sweep_axis=axis,
                        method=method,
                        method_label=label,
                        chi=chi,
                        eta=eta,
                        iteration=4,
                    ))
    return rows


def _block_size_rows():
    rows = []
    methods = (("peps_local", "local_det"), ("peps_sketch", "global_rmps_bounded"))
    for states in (1, 2, 3):
        for replicate in (0, 1):
            seed = 30_000 + 10 * states + replicate
            for method_index, (method, label) in enumerate(methods):
                rows.append(_physics_record(
                    f"block-size-{states}-{replicate}-{method}",
                    seed,
                    states,
                    (method_index + 1) * 1e-4,
                    study="block_size_sweep",
                    plot_role="physics_sweep",
                    sweep_axis="states",
                    method=method,
                    method_label=label,
                    iteration=4,
                ))
    return rows


def physics_sweep_rows():
    return _sketch_parameter_rows() + _bond_parameter_rows() + _block_size_rows()


def table_2_rows():
    references = table_2_references()
    rows = [
        _record(
            "references",
            f"table-2-exact-{hamiltonian}",
            40_000 + index,
            hamiltonian=hamiltonian,
            lx=4,
            ly=4,
            states=2,
            energies=energies,
            reference_tier="exact",
            validation_passed=True,
        )
        for index, (hamiltonian, energies) in enumerate(sorted(references.items()))
    ]
    methods = (("peps_local", "local_riemannian_ndis30"), ("peps_sketch", "global_rmps_bounded"))
    for ham_index, (hamiltonian, exact) in enumerate(sorted(references.items())):
        for chi, eta in ((4, 8), (12, 20)):
            for replicate in (0, 1):
                seed = 41_000 + 100 * ham_index + 10 * chi + replicate
                for method_index, (method, label) in enumerate(methods):
                    error = (method_index + 1) * 1e-3 / (chi + replicate + 1)
                    rows.append(_physics_record(
                        f"table-2-{hamiltonian}-{chi}-{replicate}-{method}",
                        seed,
                        2,
                        error,
                        reference_energies=exact,
                        study="dektor_reproduction",
                        plot_role="table_2",
                        dektor_panels=["table_2"],
                        hamiltonian=hamiltonian,
                        lx=4,
                        ly=4,
                        chi=chi,
                        eta=eta,
                        method=method,
                        method_label=label,
                        iteration=50,
                        published_reference_relative_tolerance=(
                            table_2_exact_relative_tolerance
                        ),
                    ))
    return rows


def dektor_convergence_rows():
    rows = []
    methods = (("peps_local", "local_riemannian_ndis30"), ("peps_sketch", "global_rmps_bounded"))
    for states in (1, 2, 3):
        for method_index, (method, label) in enumerate(methods):
            task_id = f"dektor-convergence-{states}-{method}"
            for iteration in (0, 1, 2):
                rows.append(_physics_record(
                    task_id,
                    50_000 + states,
                    states,
                    (method_index + 1) * 1e-2 / (iteration + 1),
                    study="dektor_reproduction",
                    plot_role="dektor_figure_2",
                    dektor_panels=["figure_2"],
                    hamiltonian="tfim@3.5",
                    lx=6,
                    ly=6,
                    chi=8,
                    eta=16,
                    method=method,
                    method_label=label,
                    iteration=iteration,
                ))
    return rows


def dektor_size_rows():
    rows = []
    methods = (("peps_local", "local_riemannian_ndis30"), ("peps_sketch", "global_rmps_bounded"))
    panels = {
        "figure_3": ("tfim@3", "tfim@3.5"),
        "figure_4": ("heis",),
    }
    for panel_index, (panel, hamiltonians) in enumerate(panels.items()):
        for ham_index, hamiltonian in enumerate(hamiltonians):
            for lx in (4, 6):
                for states in (1, 2):
                    seed = 60_000 + 1000 * panel_index + 100 * ham_index + 10 * lx + states
                    for method_index, (method, label) in enumerate(methods):
                        rows.append(_physics_record(
                            f"dektor-size-{panel}-{hamiltonian}-{lx}-{states}-{method}",
                            seed,
                            states,
                            (method_index + 1) * 1e-3 / lx,
                            study="dektor_reproduction",
                            plot_role=f"dektor_{panel}",
                            dektor_panels=[panel],
                            hamiltonian=hamiltonian,
                            lx=lx,
                            ly=lx,
                            chi=12,
                            eta=20 if panel == "figure_3" else 36,
                            method=method,
                            method_label=label,
                            iteration=50,
                        ))
    return rows


def _bond_sweep_rows():
    rows = []
    for ham_index, hamiltonian in enumerate(("tfim@3", "tfim@3.5", "heis")):
        for lx in (4, 6):
            for chi, eta in ((4, 8), (8, 16)):
                seed = 70_000 + 1000 * ham_index + 100 * lx + chi
                for method_index, label in enumerate(comparison_methods):
                    method = "peps_local" if label == "local_det" else "peps_sketch"
                    rows.append(_physics_record(
                        f"bond-sweep-{hamiltonian}-{lx}-{chi}-{method}",
                        seed,
                        1,
                        (method_index + 1) * 1e-3 / chi,
                        study="bond_sweep",
                        plot_role="bond_sweep",
                        hamiltonian=hamiltonian,
                        lx=lx,
                        ly=lx,
                        chi=chi,
                        eta=eta,
                        method=method,
                        method_label=label,
                        iteration=8,
                    ))
    return rows


def _hamiltonian_rows():
    rows = []
    for lx in (4, 6):
        for anisotropy in (-1.0, 0.0, 1.0):
            seed = 80_000 + 100 * lx + int(10 * (anisotropy + 1))
            for method_index, label in enumerate(comparison_methods):
                method = "peps_local" if label == "local_det" else "peps_sketch"
                rows.append(_physics_record(
                    f"xxz-{lx}-{anisotropy}-{method}",
                    seed,
                    1,
                    (method_index + 1) * 1e-3 / (lx + anisotropy + 2),
                    study="hamiltonian_robustness",
                    plot_role="hamiltonian_robustness",
                    hamiltonian=f"xxz@{anisotropy:g}",
                    lx=lx,
                    ly=lx,
                    method=method,
                    method_label=label,
                    iteration=8,
                ))
    for method_index, label in enumerate(comparison_methods):
        method = "peps_local" if label == "local_det" else "peps_sketch"
        rows.append(_physics_record(
            f"compass-{method}",
            81_000,
            1,
            (method_index + 1) * 1e-4,
            study="hamiltonian_robustness",
            plot_role="hamiltonian_robustness",
            hamiltonian="compass",
            lx=4,
            ly=4,
            method=method,
            method_label=label,
            iteration=8,
        ))
    return rows


def bond_hamiltonian_rows():
    return _bond_sweep_rows() + _hamiltonian_rows()


def extended_figure_rows():
    return (
        controlled_spectrum_rows()
        + physics_sweep_rows()
        + table_2_rows()
        + dektor_convergence_rows()
        + dektor_size_rows()
        + bond_hamiltonian_rows()
    )
