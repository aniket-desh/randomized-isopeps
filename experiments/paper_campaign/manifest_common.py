"""shared task and method specifications for the paper campaign."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rand_isopeps.campaign import derive_seed, runtime_source_fingerprint


schema_version = "paper_campaign_v1"
root_seed = 4926


def task(
    experiment: str,
    problem: Mapping,
    method: Mapping,
    seeds: Mapping,
    measurement: Mapping,
    *,
    backend: str = "numpy",
    dtype: str = "complex128",
    resources: Mapping | None = None,
    requirements: Iterable[str] = (),
    blocked: bool = False,
    blocked_reason: str | None = None,
) -> dict:
    """build one complete manifest task."""
    result = {
        "schema_version": schema_version,
        "experiment": experiment,
        "backend": backend,
        "dtype": dtype,
        "problem": dict(problem),
        "method": dict(method),
        "seeds": dict(seeds),
        "measurement": dict(measurement),
        "runtime_source_fingerprint": runtime_source_fingerprint(),
        "resources": dict(resources or {}),
        "requirements": list(requirements),
    }
    if blocked:
        result.update(blocked=True, blocked_reason=blocked_reason)
    return result


def method_seeds(problem_index: int, method_index: int, replicate: int) -> dict:
    """derive paired problem seeds and replicate-specific sketch streams."""
    problem = derive_seed(
        root_seed, "problem", int(problem_index), "replicate", int(replicate)
    )
    return {
        "root": root_seed,
        "problem": problem,
        "sketch": derive_seed(problem, "sketch", int(method_index)),
        "score": derive_seed(problem, "score", int(replicate)),
        "timing": derive_seed(problem, "timing", int(replicate)),
    }


def bundle(methods: Iterable[Mapping]) -> dict:
    """bundle paired methods after checking their labels."""
    configs = [dict(method) for method in methods]
    names = [method.get("label", method["name"]) for method in configs]
    if len(names) != len(set(names)):
        raise ValueError("method bundle labels must be unique")
    return {"name": "paired_bundle", "names": names, "configs": configs}


def bundle_seeds(problem_index: int, names: Iterable[str], replicate: int) -> dict:
    """derive one preparation seed and independent paired method streams."""
    names = tuple(names)
    problem = derive_seed(
        root_seed, "problem", int(problem_index), "replicate", int(replicate)
    )
    return {
        "root": root_seed,
        "problem": problem,
        "sketch": derive_seed(problem, "sketch", 0),
        "score": derive_seed(problem, "score", int(replicate)),
        "timing": derive_seed(problem, "timing", int(replicate)),
        "method": {
            name: {
                "sketch": derive_seed(problem, "sketch", index),
                "timing": derive_seed(problem, "timing", index),
            }
            for index, name in enumerate(names)
        },
    }


def rmps_method(
    *,
    eta: int,
    ell: int,
    chi_sk: int,
    kappa: int = 2,
    n_power: int = 0,
) -> dict:
    """return the common bounded-residual rmps configuration."""
    return {
        "name": "global_rmps",
        "eta": eta,
        "ell": ell,
        "chi_sk": chi_sk,
        "kappa": kappa,
        "n_power": n_power,
        "ndis": 0,
        "sketch_kind": "rmps",
    }


def column_methods(eta: int, *, include_dense: bool = True) -> list[dict]:
    """return the justified local and global column comparators."""
    ell = eta + 4
    methods = [
        {
            "name": "sequential_moses",
            "label": "local_det_ndis0",
            "eta": eta,
            "ndis": 0,
        },
        {
            "name": "sequential_moses_riemannian",
            "label": "local_riemannian_ndis30",
            "eta": eta,
            "ndis": 30,
            "disentangler": "riemannian_renyi",
        },
        {
            "name": "local_gaussian",
            "label": "local_rsvd2_gaussian",
            "eta": eta,
            "sketch_kind": "gaussian",
            "oversample": 4,
            "n_power": 1,
        },
        {
            "name": "local_sparsestack",
            "label": "local_rsvd2_sparsestack",
            "eta": eta,
            "sketch_kind": "sparsestack",
            "oversample": 4,
            "n_power": 1,
            "zeta": 4,
        },
    ]
    if include_dense:
        methods.extend((
            {
                "name": "global_gaussian",
                "label": "global_gaussian",
                "eta": eta,
                "ell": ell,
                "n_power": 0,
                "materialized_only": True,
            },
            {
                "name": "global_rademacher",
                "label": "global_rademacher",
                "eta": eta,
                "ell": ell,
                "n_power": 0,
                "materialized_only": True,
            },
            {
                "name": "global_sparsestack",
                "label": "global_sparsestack",
                "eta": eta,
                "ell": ell,
                "n_power": 0,
                "zeta": 4,
                "materialized_only": True,
            },
        ))
    methods.extend((
        {**rmps_method(eta=eta, ell=ell, chi_sk=8), "label": "global_rmps_bounded"},
        {
            **rmps_method(eta=eta, ell=ell, chi_sk=1),
            "name": "global_kron",
            "label": "global_kron",
            "sketch_kind": "kron",
        },
    ))
    return methods


def global_column_methods(eta: int) -> list[dict]:
    """return global methods valid for dense controlled-spectrum matrices."""
    return [
        method for method in column_methods(eta)
        if str(method["name"]).startswith("global_")
    ]


def hamiltonian(name: str, parameter: float | None = None) -> str:
    """encode one Hamiltonian using the executor grammar."""
    if name == "tfim":
        return f"tfim@{float(parameter):g}"
    if name == "xxz":
        return f"xxz@{float(parameter):g}"
    if name == "heisenberg":
        return "heis"
    return name
