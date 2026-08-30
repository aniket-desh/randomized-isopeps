from experiments.paper_campaign.build_manifests import builders
from rand_isopeps.campaign import finalize_task


def test_all_campaign_families_build_unique_tasks():
    assert set(builders) == {
        "gaussian_limit",
        "column_moves",
        "isometry",
        "physics",
        "gpu_pilot",
        "gpu_crossover",
        "references",
    }
    for builder in builders.values():
        tasks = [finalize_task(task) for task in builder()]
        identifiers = [task["task_id"] for task in tasks]
        assert tasks
        assert len(identifiers) == len(set(identifiers))


def test_prepared_columns_bundle_paired_methods():
    tasks = builders["column_moves"]()
    assert all(task["method"]["name"] == "paired_bundle" for task in tasks)
    assert all(task["method"]["names"] for task in tasks)
    assert all("method" in task["seeds"] for task in tasks)
    assert {task["problem"]["source"] for task in tasks} == {"physical", "synthetic"}
    assert any(task["problem"].get("state") == "random_raw" for task in tasks)
    assert all("lx" in task["problem"] for task in tasks)
    assert all(task["dtype"] == "float64" for task in tasks)
    assert all(
        "ly" in task["problem"]
        for task in tasks
        if task["problem"]["source"] == "physical"
    )


def test_optimized_moses_comparator_uses_supported_disentangler():
    task = next(
        task
        for task in builders["column_moves"]()
        if task["measurement"]["study"] == "method_comparison"
    )
    method = next(
        method
        for method in task["method"]["configs"]
        if method["name"] == "sequential_moses_riemannian"
    )
    assert method["ndis"] == 30
    assert method["disentangler"] == "riemannian_renyi"


def test_parameter_sweep_changes_one_factor_at_a_time():
    task = next(
        task
        for task in builders["column_moves"]()
        if task["measurement"]["study"] == "one_at_a_time"
    )
    baseline = task["measurement"]["baseline"]
    keys = ("eta", "ell", "chi_sk", "kappa", "n_power")
    for method in task["method"]["configs"]:
        differences = sum(method[key] != baseline[key] for key in keys)
        assert differences <= 1


def test_multistate_physics_tasks_use_shared_block_moves():
    tasks = builders["physics"]()
    multistate = [task for task in tasks if task["problem"]["states"] > 1]
    assert multistate
    assert all(not task.get("blocked", False) for task in multistate)
    assert all("block_peps" in task["requirements"] for task in multistate)
    assert any(
        task["problem"]["states"] == 1 and not task.get("blocked") for task in tasks
    )
    assert all("stages" in task["problem"] for task in tasks)
    assert all(isinstance(task["problem"]["hamiltonian"], str) for task in tasks)
    assert all(
        task["dtype"]
        == (
            "complex128"
            if task["problem"]["initialization"] == "random_product"
            else "float64"
        )
        for task in tasks
    )
    paper_tasks = [
        task
        for task in tasks
        if task["problem"]["study"] not in {"correctness_ladder", "block_correctness"}
        and "reference_energies" not in task["problem"]
    ]
    assert all("reference_artifact" in task["requirements"] for task in paper_tasks)
    dektor = [
        task for task in tasks if task["problem"]["study"] == "dektor_reproduction"
    ]
    assert all(task["backend"] == "numpy" for task in dektor)
    assert all("block_peps" in task["requirements"] for task in dektor)
    dektor_local = [
        task for task in dektor if task["method"]["name"] == "peps_local"
    ]
    assert all(task["method"]["ndis"] == 30 for task in dektor_local)
    assert all(
        task["method"]["disentangler"] == "riemannian_renyi"
        for task in dektor_local
    )
    assert all(
        task["method"]["label"] == "local_riemannian_ndis30"
        for task in dektor_local
    )
    assert all(
        task["backend"] == "cupy"
        for task in paper_tasks
        if task["method"]["name"] == "peps_sketch"
        and task["problem"]["states"] == 1
        and task["problem"]["study"] != "dektor_reproduction"
    )
    p_sweeps = [
        task
        for task in tasks
        if task["problem"]["study"]
        in {"sketch_parameter_sweep", "block_size_sweep"}
    ]
    assert all(task["problem"]["trotter_order"] == 1 for task in p_sweeps)


def test_dektor_p_variants_share_nested_initialization_streams():
    tasks = [
        task
        for task in builders["physics"]()
        if task["problem"]["study"] == "dektor_reproduction"
    ]
    groups = {}
    for task in tasks:
        problem = task["problem"]
        key = (
            problem["lx"],
            problem["ly"],
            problem["hamiltonian"],
            problem["chi"],
            problem["eta"],
        )
        groups.setdefault(key, []).append(task)

    paired = 0
    cell_problem_streams = []
    for group in groups.values():
        by_states = {}
        for task in group:
            by_states.setdefault(task["problem"]["states"], []).append(task)
        parent = max(by_states)
        assert {
            task["problem"]["initialization_parent_states"] for task in group
        } == {parent}
        problem_streams = [
            {task["seeds"]["problem"] for task in state_tasks}
            for state_tasks in by_states.values()
        ]
        assert all(stream == problem_streams[0] for stream in problem_streams)
        cell_problem_streams.append(problem_streams[0])
        method_streams = []
        for method_name in ("peps_local", "peps_sketch"):
            sketch_streams = [
                {
                    task["seeds"]["sketch"]
                    for task in state_tasks
                    if task["method"]["name"] == method_name
                }
                for state_tasks in by_states.values()
            ]
            assert all(stream == sketch_streams[0] for stream in sketch_streams)
            method_streams.append(sketch_streams[0])
        assert method_streams[0].isdisjoint(method_streams[1])
        paired += len(by_states) > 1
    assert paired
    assert len(set().union(*cell_problem_streams)) == sum(
        map(len, cell_problem_streams)
    )


def test_block_size_sweep_shares_parent_and_paired_streams():
    tasks = [
        task
        for task in builders["physics"]()
        if task["problem"]["study"] == "block_size_sweep"
    ]
    by_states = {}
    for task in tasks:
        by_states.setdefault(task["problem"]["states"], []).append(task)
    assert set(by_states) == {1, 2, 3}
    assert {
        task["problem"]["initialization_parent_states"] for task in tasks
    } == {3}
    problem_streams = [
        {task["seeds"]["problem"] for task in state_tasks}
        for state_tasks in by_states.values()
    ]
    assert all(stream == problem_streams[0] for stream in problem_streams)
    method_streams = []
    for method_name in ("peps_local", "peps_sketch"):
        sketch_streams = [
            {
                task["seeds"]["sketch"]
                for task in state_tasks
                if task["method"]["name"] == method_name
            }
            for state_tasks in by_states.values()
        ]
        assert all(stream == sketch_streams[0] for stream in sketch_streams)
        method_streams.append(sketch_streams[0])
    assert method_streams[0].isdisjoint(method_streams[1])


def test_correctness_manifests_label_the_actual_trotter_order():
    tasks = builders["physics"]()
    single = [
        task for task in tasks if task["problem"]["study"] == "correctness_ladder"
    ]
    block = [
        task for task in tasks if task["problem"]["study"] == "block_correctness"
    ]

    assert any(task["method"]["name"] == "dense_strang" for task in single)
    assert all(
        task["problem"]["trotter_order"] == 2
        for task in single
        if task["method"]["name"] == "dense_strang"
    )
    assert any(task["method"]["name"] == "dense_first_order" for task in block)
    assert not any(task["method"]["name"] == "dense_strang" for task in block)
    assert all(task["problem"]["trotter_order"] == 1 for task in block)


def test_physics_manifest_dtypes_match_representative_initializers():
    from rand_isopeps.campaign.physics_block import initial_block
    from rand_isopeps.campaign.physics_common import site_vectors
    from rand_isopeps.campaign.physics_single import initial_peps

    product = initial_peps(
        {
            "problem": {"lx": 2, "ly": 2, "initialization": "random_product"},
            "seeds": {"problem": 17},
            "backend": "numpy",
        },
        site_vectors(2, 2, 17),
    )
    block = initial_block(
        {
            "lx": 2,
            "ly": 2,
            "states": 2,
            "bond": 2,
            "chi": 2,
            "eta": 2,
            "study": "block_correctness",
        },
        19,
    )

    assert {tensor.data.dtype.name for tensor in product.tensors} == {"complex128"}
    assert {tensor.data.dtype.name for tensor in block.peps.tensors} == {"float64"}


def test_gpu_pilot_pairs_backends_and_declares_cupy():
    tasks = builders["gpu_pilot"]()
    assert {task["backend"] for task in tasks} == {"numpy", "cupy"}
    assert all(
        ("cupy" in task["requirements"]) == (task["backend"] == "cupy")
        for task in tasks
    )
    assert all(task["method"]["name"] != "paired_bundle" for task in tasks)
    column_tasks = [task for task in tasks if task["experiment"] == "gpu_pilot"]
    assert all("sketch" in task["seeds"] for task in column_tasks)


def test_gaussian_limit_uses_executor_fields():
    tasks = builders["gaussian_limit"]()
    embeddings = [task for task in tasks if task["problem"]["kind"] == "sketch_embedding"]
    variance = [task for task in tasks if task["problem"]["kind"] == "walsh_variance"]
    nystrom = [task for task in tasks if task["problem"]["kind"] == "walsh_nystrom"]
    assert len(embeddings) == 240
    assert len(variance) == 52
    assert len(nystrom) == 128
    assert all(task["measurement"]["replicates"] == 60 for task in embeddings)
    assert all("lx" in task["problem"] for task in embeddings)
    assert {task["method"]["name"] for task in embeddings} == {
        "global_gaussian",
        "rmps",
    }
    assert all(task["measurement"]["replicates"] == 20 for task in variance + nystrom)
    assert {task["dtype"] for task in variance + nystrom} == {"float64"}


def test_reference_builder_uses_exact_small_cells_and_paired_large_bonds():
    tasks = builders["references"]()
    small = [task for task in tasks if task["problem"]["lx"] * task["problem"]["ly"] <= 16]
    large = [task for task in tasks if task not in small]
    assert small and large
    assert all(task["method"]["name"] == "exact_diagonalization" for task in small)
    assert all(task["method"]["name"] == "dmrg_reference" for task in large)
    assert all(task["dtype"] == "float64" for task in small)
    assert all(task["dtype"] == "complex128" for task in large)
    table_2 = {
        task["problem"]["hamiltonian"]: int(task["problem"]["states"])
        for task in small
        if task["problem"]["hamiltonian"] in {
            "tfim@1",
            "tfim@2",
            "tfim@3",
            "tfim@3.5",
        }
    }
    assert table_2 == {
        "tfim@1": 2,
        "tfim@2": 2,
        "tfim@3": 2,
        "tfim@3.5": 3,
    }


def test_large_references_use_strict_energy_orthogonality_contract():
    references = [
        task
        for task in builders["references"]()
        if task["problem"].get("reference_tier") == "paper_energy"
    ]
    assert references
    assert not any(
        task["problem"].get("reference_tier") == "residual_validation"
        for task in builders["references"]()
    )
    assert all(
        task["measurement"]["validation_contract"]
        == "nested_bond_energy_orthogonality"
        and task["measurement"]["residual_required"] is False
        for task in references
    )
    projected = [
        task for task in references
        if task["problem"]["states"] > 1
        and not task["method"]["target_sectors"]
    ]
    assert projected
    assert all(
        task["method"]["projector_state_bond"] == 64
        and task["method"]["projector_state_tolerance"] == 1e-4
        for task in projected
    )
