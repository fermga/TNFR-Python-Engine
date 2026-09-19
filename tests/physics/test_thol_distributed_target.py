"""Fixed original-profile accounting for one finite distributed UM response."""

import hashlib
import json
from fractions import Fraction
from unittest.mock import patch

import pytest

from benchmarks.thol_pressure_feedback import _payload


@pytest.fixture(scope="module")
def execution():
    import benchmarks.thol_distributed_target as benchmark

    prepare = benchmark.prepare_distributed_transport_support
    graphs = []

    def record_fresh_source(*args, **kwargs):
        graph, prefix = prepare(*args, **kwargs)
        graphs.append(graph)
        return graph, prefix

    with patch.object(
        benchmark, "prepare_distributed_transport_support", record_fresh_source
    ):
        study = benchmark.run_study()
    return study, graphs


@pytest.fixture(scope="module")
def study(execution):
    return execution[0]


@pytest.fixture(scope="module")
def branches(study):
    return {branch["branch"]: branch for branch in study["branches"]}


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _laplacian(edges, values):
    output = [Fraction(0)] * len(values)
    for i, j, weight in edges:
        output[i] += weight * (values[i] - values[j])
    return tuple(output)


def _energy(edges, values):
    return (
        sum(
            (weight * (values[i] - values[j]) ** 2 for i, j, weight in edges),
            Fraction(0),
        )
        / 4
    )


def _project(metric, values):
    mean = _dot(metric, values) / sum(metric, Fraction(0))
    return tuple(value - mean for value in values)


def _assert_reference(reference):
    source = reference["source"]
    strengths = [Fraction(0)] * len(source["nodes"])
    for i, _, weight in source["conductance"]:
        strengths[i] += weight
    metric = tuple(d / nu for d, nu in zip(strengths, source["capacity"], strict=True))
    assert all(value > 0 for value in metric)
    assert reference["strengths"] == tuple(strengths)
    assert reference["metric_weights"] == metric
    drift = _dot(strengths, reference["forcing"]) / sum(metric, Fraction(0))
    assert reference["mean_drift"] == drift
    assert reference["compatibility_residual"] == _dot(strengths, reference["forcing"])
    z = reference["relative_profile"]
    assert _dot(metric, z) == reference["profile_center_residual"] == 0
    left = tuple(
        reference["epi_weight"] * value
        for value in _laplacian(source["conductance"], z)
    )
    right = tuple(
        d * f - drift * h
        for d, f, h in zip(
            strengths,
            reference["forcing"],
            metric,
            strict=True,
        )
    )
    assert left == right
    assert reference["profile_residual"] == (Fraction(0),) * len(z)
    assert reference["has_zero_pressure_equilibrium"] == (drift == 0)
    reached, pending = {0}, [0]
    while pending:
        node = pending.pop()
        for i, j, weight in source["conductance"]:
            if i == node and weight > 0 and j not in reached:
                reached.add(j)
                pending.append(j)
    assert reached == set(range(len(source["nodes"])))


def _assert_pattern(reference, pattern):
    assert pattern["nodes"] == reference["source"]["nodes"]
    metric = reference["metric_weights"]
    x = pattern["epi"]
    mean = _dot(metric, x) / sum(metric, Fraction(0))
    error = tuple(
        value - mean - z
        for value, z in zip(
            x,
            reference["relative_profile"],
            strict=True,
        )
    )
    assert pattern["mean"] == mean
    assert pattern["relative_error"] == error
    assert (
        pattern["error_variance"]
        == _dot(metric, tuple(value**2 for value in error)) / 2
    )
    assert pattern["error_dirichlet_energy"] == _energy(
        reference["source"]["conductance"], error
    )


def _assert_target(target, original):
    assert target["target_reference"] == original
    current = target["reference"]
    _assert_reference(current)
    assert target["pattern"]["epi"] == target["state"]["snapshot"]["epi"]
    _assert_pattern(original, target["pattern"])
    _assert_pattern(original, target["limiting_pattern"])
    assert target["limiting_pattern"]["epi"] == current["relative_profile"]
    metric = original["metric_weights"]
    source = current["source"]
    nu, d = source["capacity"], current["strengths"]

    def action(values):
        return tuple(
            current["epi_weight"] * capacity * gradient / strength
            for capacity, gradient, strength in zip(
                nu, _laplacian(source["conductance"], values), d, strict=True
            )
        )

    az = action(original["relative_profile"])
    target_rate = tuple(
        capacity * f - value
        for capacity, f, value in zip(
            nu,
            current["forcing"],
            az,
            strict=True,
        )
    )
    residual = _project(metric, target_rate)
    assert target["target_rate"] == target_rate
    assert target["compatibility_residual"] == residual
    assert target["target_compatible"] == (not any(residual))
    assert target["profile_identity_residual"] == (0,) * len(residual)
    assert (
        _project(metric, action(target["limiting_pattern"]["relative_error"]))
        == residual
    )
    pressure_channels = dict(target["pressure_channels"])
    assert tuple(pressure_channels) == ("epi", "phase", "vf", "topo")
    assert pressure_channels["epi"] == tuple(
        -value / capacity
        for value, capacity in zip(
            az,
            nu,
            strict=True,
        )
    )
    assert (
        tuple(
            sum(
                (pressure_channels[name][i] for name in ("phase", "vf", "topo")),
                Fraction(0),
            )
            for i in range(len(nu))
        )
        == current["forcing"]
    )
    projected = tuple(
        (
            name,
            _project(
                metric,
                tuple(
                    capacity * p
                    for capacity, p in zip(
                        nu,
                        values,
                        strict=True,
                    )
                ),
            ),
        )
        for name, values in pressure_channels.items()
    )
    assert target["projected_rate_channels"] == projected
    gram = tuple(
        tuple(
            _dot(metric, tuple(a * b for a, b in zip(left, right, strict=True)))
            for _, right in projected
        )
        for _, left in projected
    )
    assert target["channel_gram"] == gram
    energy = _dot(metric, tuple(value**2 for value in residual)) / 2
    assert target["compatibility_energy"] == energy
    assert energy == sum((sum(row, Fraction(0)) for row in gram), Fraction(0)) / 2
    assert target["channel_energy_identity_residual"] == 0
    error = target["pattern"]["relative_error"]
    homogeneous = -_dot(
        metric,
        tuple(
            value * rate
            for value, rate in zip(
                error,
                action(error),
                strict=True,
            )
        ),
    )
    source_rate = _dot(
        metric, tuple(value * rate for value, rate in zip(error, residual, strict=True))
    )
    state = target["state"]
    x = state["snapshot"]["epi"]
    model = tuple(
        capacity * f - drift
        for capacity, f, drift in zip(
            nu,
            current["forcing"],
            action(x),
            strict=True,
        )
    )
    model_energy_rate = _dot(
        metric, tuple(value * rate for value, rate in zip(error, model, strict=True))
    )
    pressure_defect = tuple(
        p - rate / capacity
        for p, rate, capacity in zip(
            state["snapshot"]["stored_pressure"],
            model,
            nu,
            strict=True,
        )
    )
    defect_rate = _dot(
        metric,
        tuple(
            value * capacity * defect
            for value, capacity, defect in zip(
                error,
                nu,
                pressure_defect,
                strict=True,
            )
        ),
    )
    assert target["model_rate"] == model
    assert state["pressure_defect"] == pressure_defect
    assert target["homogeneous_energy_rate"] == homogeneous
    assert target["target_source_energy_rate"] == source_rate
    assert target["model_energy_rate"] == model_energy_rate == homogeneous + source_rate
    assert target["energy_rate_identity_residual"] == 0
    assert target["stored_pressure_energy_rate_defect"] == defect_rate
    assert target["stored_nodal_energy_rate"] == model_energy_rate + defect_rate


def _assert_step(step):
    reference = step["reference"]
    before, after = step["before"], step["after"]
    metric = reference["metric_weights"]
    mass = sum(metric, Fraction(0))
    h = step["dt"]
    x = before["snapshot"]["epi"]
    y = after["snapshot"]["epi"]
    nu = reference["source"]["capacity"]
    pressure = before["snapshot"]["stored_pressure"]
    for state in (before, after):
        snapshot = state["snapshot"]
        assert snapshot["nodes"] == reference["source"]["nodes"]
        assert snapshot["conductance"] == reference["source"]["conductance"]
        assert snapshot["capacity"] == nu
        mean = _dot(metric, snapshot["epi"]) / mass
        error = tuple(
            value - mean - target
            for value, target in zip(
                snapshot["epi"],
                reference["relative_profile"],
                strict=True,
            )
        )
        modeled = tuple(
            -reference["epi_weight"] * gradient / d + f
            for gradient, d, f in zip(
                _laplacian(snapshot["conductance"], snapshot["epi"]),
                reference["strengths"],
                reference["forcing"],
                strict=True,
            )
        )
        assert state["mean"] == mean
        assert state["relative_error"] == error
        assert state["modeled_pressure"] == modeled
        assert state["pressure_defect"] == tuple(
            p - q
            for p, q in zip(
                snapshot["stored_pressure"],
                modeled,
                strict=True,
            )
        )
    defect = tuple(
        final - initial - h * capacity * p
        for final, initial, capacity, p in zip(y, x, nu, pressure, strict=True)
    )
    measured = (
        _dot(
            metric, tuple(final - initial for final, initial in zip(y, x, strict=True))
        )
        / mass
    )
    model = h * reference["mean_drift"]
    pressure_term = h * _dot(reference["strengths"], before["pressure_defect"]) / mass
    rounding = _dot(metric, defect) / mass
    assert step["mean_change"] == measured
    assert step["mean_model_change"] == model
    assert step["mean_pressure_defect"] == pressure_term
    assert step["mean_step_defect"] == rounding
    assert (
        step["mean_identity_residual"]
        == measured - model - pressure_term - rounding
        == 0
    )
    assert step["support_budget"]["state_defect"] == defect
    combined = tuple(
        h * capacity * p + residual
        for capacity, p, residual in zip(
            nu,
            before["pressure_defect"],
            defect,
            strict=True,
        )
    )
    centered = _project(metric, combined)
    assert step["relative_energy_budget"]["state_defect"] == centered
    assert step["relative_recurrence_residual"] == (0,) * len(x)
    for key in ("support_budget", "relative_energy_budget"):
        budget = step[key]
        assert budget["identity_residual"] == 0
        assert budget["energy_change"] == (
            budget["drift_term"] + budget["quadratic_term"] + budget["defect_term"]
        )


def test_branches_replay_independent_graphs_and_the_identical_causal_baseline(
    execution, branches
):
    study, graphs = execution
    assert len(graphs) == 2 and graphs[0] is not graphs[1]
    assert tuple(branches) == ("no_event", "all_node_um")
    assert study["common_causal_baseline_reproduced"]
    left, right = branches.values()
    for key in (
        "prefix",
        "original_reference",
        "initial_target",
        "baseline_flow",
        "baseline_target",
        "baseline_steps",
        "before_optional_event",
        "common_source",
    ):
        assert left[key] == right[key]
    assert left["before_optional_event"]["time"] == 1.0
    assert len(left["before_optional_event"]["nodes"]) == 16


@pytest.mark.parametrize("name", ("no_event", "all_node_um"))
def test_original_reference_is_frozen_at_the_first_refreshed_postgrowth_source(
    branches, name
):
    branch = branches[name]
    original = branch["original_reference"]
    source = branch["prefix"]["coupling"]["refreshed_forcing"]["observation"]
    assert original["source"] == source["snapshot"]
    assert original["forcing"] == source["forcing"]
    assert original["epi_weight"] == source["epi_weight"]
    assert branch["original_reference_frozen_time"] == 0.5
    assert branch["original_reference_frozen_before_baseline_flow"]
    _assert_reference(original)
    encoded = json.dumps(
        _payload(original), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    assert branch["original_reference_sha256"] == digest
    assert branch["initial_target"]["pattern"]["epi"] == original["source"]["epi"]
    assert branch["initial_target"]["state"]["snapshot"] == source["snapshot"]
    assert (
        branch["baseline_target"]["state"]["snapshot"]
        == branch["baseline_flow"]["after_forcing"]["observation"]["snapshot"]
    )
    assert (
        branch["endpoint_target"]["state"]["snapshot"]
        == branch["continuation_flow"]["after_forcing"]["observation"]["snapshot"]
    )
    for key in (
        "initial_target",
        "baseline_target",
        "post_event_target",
        "endpoint_target",
    ):
        _assert_target(branch[key], original)


@pytest.mark.parametrize("name", ("no_event", "all_node_um"))
def test_both_finite_partitions_keep_declared_current_models_and_exact_mean_defect_budgets(
    branches, name
):
    branch = branches[name]
    for flow_key, step_key, start in (
        ("baseline_flow", "baseline_steps", 0.5),
        ("continuation_flow", "continuation_steps", 1.0),
    ):
        flow = branch[flow_key]
        steps = branch[step_key]
        assert flow["before"]["time"] == start
        assert flow["after"]["time"] == start + 0.5
        assert flow["partition"]["segment_durations"] == (0.25, 0.25)
        assert tuple(boundary["time"] for boundary in flow["boundaries"]) == (
            start,
            start + 0.25,
            start + 0.5,
        )
        assert len(steps) == len(flow["segments"]) == 2
        assert flow["all_segment_binary64_replays_identified"]
        current = (
            branch["original_reference"]
            if flow_key == "baseline_flow"
            else (branch["post_event_target"]["reference"])
        )
        for segment, step in zip(flow["segments"], steps, strict=True):
            assert step["before"]["snapshot"] == segment["before_support"]
            assert step["after"]["snapshot"] == segment["after_support"]
            assert step["reference"] == current
            assert step["dt"] == Fraction(1, 4)
            assert not segment["clipping_applied"]
            _assert_step(step)
    assert branch["continuation_flow"]["before"] == branch["after_optional_event"]
    assert branch["endpoint"] == branch["continuation_flow"]["after"]
    assert branch["endpoint"]["time"] == 1.5


def test_no_event_control_preserves_the_actual_common_endpoint_and_model(branches):
    branch = branches["no_event"]
    assert branch["event"] is None
    assert branch["before_optional_event"] == branch["after_optional_event"]
    assert branch["baseline_target"] == branch["post_event_target"]
    assert branch["endpoint_target"]["reference"] == branch["original_reference"]
    for key in (
        "initial_target",
        "baseline_target",
        "post_event_target",
        "endpoint_target",
    ):
        assert branch[key]["target_compatible"]
        assert branch[key]["compatibility_energy"] == 0


def test_all_current_nodes_receive_one_actual_admitted_default_um_stage(branches):
    branch = branches["all_node_um"]
    coupling = branch["event"]["coupling"]
    source = branch["before_optional_event"]
    assert source == coupling["before"] == branch["common_source"]["state"]
    assert len(source["nodes"]) == 16
    assert coupling["targets"] == coupling["node_sample_used"] == source["nodes"]
    assert coupling["sampling_refresh_executed"]
    assert coupling["functional_links"] and coupling["bidirectional"]
    assert coupling["candidate_limit"] == 0 and coupling["candidate_mode"] == "sample"
    assert coupling["stage_result"]["schedule"] == "two_phase_jacobi"
    assert coupling["stage_result"]["nodes_processed"] == 16
    assert tuple(row["node"] for row in coupling["admissions"]) == source["nodes"]
    assert all(
        row["allowed"] and row["candidate"] == "UM" for row in coupling["admissions"]
    )
    assert coupling["after_raw"]["nodes"] == source["nodes"]
    assert coupling["after_raw"]["epi"] == source["epi"]
    assert coupling["after_raw"]["time"] == source["time"] == 1.0
    proposal = coupling["kernel_proposal"]
    assert proposal["targets"] == source["nodes"]
    assert len(proposal["target_proposals"]) == 16
    expected = {
        field: list(source[field]) for field in ("phase", "capacity", "pressure")
    }
    for update in proposal["node_updates"]:
        index = source["nodes"].index(update["node"])
        for key, field in (
            ("theta_after", "phase"),
            ("vf_after", "capacity"),
            ("dnfr_after", "pressure"),
        ):
            if update[key] is not None:
                expected[field][index] = update[key]
    for field, values in expected.items():
        assert coupling["after_raw"][field] == tuple(values)
    assert {
        frozenset((u, v)): data["weight"] for u, v, data in coupling["new_edges"]
    } == {
        frozenset((edge["left"], edge["right"])): edge["weight"]
        for edge in proposal["edges"]
    }
    assert len(coupling["new_edges"]) == 8
    assert len(coupling["after_raw"]["edges"]) == 32
    assert coupling["support_inventory"]["directed_unique_support_entries"] == 64
    for node in source["nodes"]:
        assert coupling["after_raw"]["glyph_history"][node] == source["glyph_history"][
            node
        ] + ("UM",)


def test_zero_epi_event_cannot_be_counted_as_instantaneous_old_target_recovery(
    branches,
):
    branch = branches["all_node_um"]
    event = branch["event"]["exact_forced_event"]
    assert event["before_reference"] == branch["original_reference"]
    assert event["after_reference"] == branch["post_event_target"]["reference"]
    assert (
        event["before"]["snapshot"]
        == branch["baseline_flow"]["after_forcing"]["observation"]["snapshot"]
    )
    assert (
        event["after"]["snapshot"]
        == branch["event"]["coupling"]["refreshed_forcing"]["observation"]["snapshot"]
    )
    assert event["epi_jump"] == event["centered_epi_jump"] == (0,) * 16
    assert event["mean_epi_jump"] == 0
    assert (
        branch["post_event_target"]["pattern"] == branch["baseline_target"]["pattern"]
    )
    assert event["midpoint_pattern"] == branch["baseline_target"]["pattern"]
    for field in ("variance_jump_budget", "dirichlet_jump_budget"):
        assert all(value == 0 for value in event[field].values())
    before, after = event["before"], event["after"]
    first, second = event["before_reference"], event["after_reference"]
    u0, u1 = before["relative_error"], after["relative_error"]
    shift = tuple(new - old for old, new in zip(u0, u1, strict=True))
    h0, h1 = first["metric_weights"], second["metric_weights"]
    variance = event["variance_reset_budget"]
    assert (
        variance["metric_term"]
        == _dot(
            tuple(new - old for old, new in zip(h0, h1, strict=True)),
            tuple(value**2 for value in u0),
        )
        / 2
    )
    assert variance["reference_cross_term"] == _dot(
        h1, tuple(a * b for a, b in zip(u0, shift, strict=True))
    )
    assert (
        variance["reference_quadratic_term"]
        == _dot(h1, tuple(value**2 for value in shift)) / 2
    )
    old_edges, new_edges = (
        first["source"]["conductance"],
        second["source"]["conductance"],
    )
    dirichlet = event["dirichlet_reset_budget"]
    assert dirichlet["metric_term"] == _energy(new_edges, u0) - _energy(old_edges, u0)
    assert dirichlet["reference_cross_term"] == _dot(_laplacian(new_edges, u0), shift)
    assert dirichlet["reference_quadratic_term"] == _energy(new_edges, shift)
    for budget in (variance, dirichlet):
        assert budget["energy_change"] == (
            budget["metric_term"]
            + budget["reference_cross_term"]
            + budget["reference_quadratic_term"]
        )
        assert budget["identity_residual"] == 0
    assert event["mean_reweighting"] == after["mean"] - before["mean"]
    assert event["mean_change"] == event["mean_reweighting"]
    assert event["mean_identity_residual"] == 0
    assert event["error_identity_residual"] == (0,) * 16
    assert event["variance_change"] == variance["energy_change"]
    assert event["dirichlet_change"] == dirichlet["energy_change"]
    assert (
        event["variance_identity_residual"] == event["dirichlet_identity_residual"] == 0
    )


def test_signed_channel_allocation_keeps_cross_terms_and_the_same_target_metric(
    branches,
):
    branch = branches["all_node_um"]
    before, after = branch["baseline_target"], branch["post_event_target"]
    old, new = dict(before["projected_rate_channels"]), dict(
        after["projected_rate_channels"]
    )
    metric = branch["original_reference"]["metric_weights"]
    midpoint = tuple(
        (a + b) / 2
        for a, b in zip(
            before["compatibility_residual"],
            after["compatibility_residual"],
            strict=True,
        )
    )
    expected = tuple(
        (
            name,
            _dot(
                metric,
                tuple(
                    mid * (b - a)
                    for mid, a, b in zip(
                        midpoint,
                        old[name],
                        new[name],
                        strict=True,
                    )
                ),
            ),
        )
        for name in old
    )
    allocation = branch["event"]["signed_target_channel_change"]
    assert allocation["channel_contributions"] == expected
    change = after["compatibility_energy"] - before["compatibility_energy"]
    assert allocation["compatibility_energy_change"] == change
    assert change == sum((value for _, value in expected), Fraction(0))
    assert allocation["identity_residual"] == 0
    assert "not separately executed ablations" in allocation["scope"]


def test_same_state_rate_change_keeps_generator_target_and_realization_terms_separate(
    branches,
):
    branch = branches["all_node_um"]
    before, after = branch["baseline_target"], branch["post_event_target"]
    assert before["pattern"] == after["pattern"]
    metric = branch["original_reference"]["metric_weights"]
    error = before["pattern"]["relative_error"]

    def actual_rate(target):
        snapshot = target["state"]["snapshot"]
        return _dot(
            metric,
            tuple(
                u * nu * p
                for u, nu, p in zip(
                    error,
                    snapshot["capacity"],
                    snapshot["stored_pressure"],
                    strict=True,
                )
            ),
        )

    total = actual_rate(after) - actual_rate(before)
    changes = {
        field: after[field] - before[field]
        for field in (
            "homogeneous_energy_rate",
            "target_source_energy_rate",
            "stored_pressure_energy_rate_defect",
        )
    }
    assert total == sum(changes.values(), Fraction(0))
    assert (
        total == after["stored_nodal_energy_rate"] - before["stored_nodal_energy_rate"]
    )
    retained = branch["event"]["same_state_rate_change"]
    assert retained["before_stored_nodal_energy_rate"] == actual_rate(before)
    assert retained["after_stored_nodal_energy_rate"] == actual_rate(after)
    assert retained["stored_nodal_energy_rate_change"] == total
    for name, value in changes.items():
        assert retained[name + "_change"] == value
    assert retained["identity_residual"] == 0


@pytest.mark.parametrize("name", ("no_event", "all_node_um"))
def test_forcing_is_captured_from_channels_and_stays_distinct_from_stored_pressure(
    branches, name
):
    branch = branches[name]
    pairs = [
        (
            branch["prefix"]["coupling"]["raw_forcing"],
            branch["prefix"]["coupling"]["refreshed_forcing"],
        ),
    ]
    if branch["event"] is not None:
        coupling = branch["event"]["coupling"]
        pairs.append((coupling["raw_forcing"], coupling["refreshed_forcing"]))
    for raw, fresh in pairs:
        first, second = raw["observation"], fresh["observation"]
        for field in (
            "forcing",
            "phase",
            "phase_gradient",
            "epi_weight",
            "normalized_weights",
            "full_kernel_pressure",
            "kernel_pressure_defect",
        ):
            assert first[field] == second[field]
        assert (
            first["snapshot"]["stored_pressure"]
            != second["snapshot"]["stored_pressure"]
        )
        for record in (raw, fresh):
            observation = record["observation"]
            assert (
                tuple(
                    sum((vector[i] for _, vector in record["components"]), Fraction(0))
                    for i in range(16)
                )
                == observation["forcing"]
            )
            modeled = tuple(
                observation["epi_weight"] * gradient + force
                for gradient, force in zip(
                    observation["snapshot"]["epi_gradient"],
                    observation["forcing"],
                    strict=True,
                )
            )
            assert (
                tuple(
                    actual - expected
                    for actual, expected in zip(
                        observation["full_kernel_pressure"],
                        modeled,
                        strict=True,
                    )
                )
                == observation["kernel_pressure_defect"]
            )
            assert (
                tuple(
                    actual - kernel
                    for actual, kernel in zip(
                        observation["snapshot"]["stored_pressure"],
                        observation["full_kernel_pressure"],
                        strict=True,
                    )
                )
                == observation["stored_pressure_residual"]
            )
    for key in ("baseline_capture_checks", "continuation_capture_checks"):
        assert all(value for field, value in branch[key].items() if field != "scope")


def test_observed_short_term_error_decrease_does_not_erase_new_target_incompatibility(
    branches,
):
    baseline = branches["no_event"]
    changed = branches["all_node_um"]
    initial = baseline["initial_target"]["pattern"]["error_variance"]
    common = baseline["baseline_target"]["pattern"]["error_variance"]
    no_event = baseline["endpoint_target"]["pattern"]["error_variance"]
    with_event = changed["endpoint_target"]["pattern"]["error_variance"]
    assert initial > common > with_event > no_event > 0
    assert not baseline["comparison"]["initial_target_error_is_zero"]
    assert baseline["endpoint_target"]["target_compatible"]
    assert not changed["post_event_target"]["target_compatible"]
    assert not changed["endpoint_target"]["target_compatible"]
    assert changed["endpoint_target"]["compatibility_energy"] > 0
    assert (
        changed["post_event_target"]["reference"]["relative_profile"]
        != changed["original_reference"]["relative_profile"]
    )
    assert (
        changed["endpoint_target"]["target_reference"] == baseline["original_reference"]
    )
    for branch in branches.values():
        comparison = branch["comparison"]
        assert comparison["baseline_fixed_target_variance_change"] == common - initial
        assert comparison["continuation_fixed_target_variance_change"] == (
            branch["endpoint_target"]["pattern"]["error_variance"]
            - branch["post_event_target"]["pattern"]["error_variance"]
        )
        assert comparison["original_metric_mean_change"] == (
            branch["endpoint_target"]["pattern"]["mean"]
            - branch["initial_target"]["pattern"]["mean"]
        )


def test_finite_json_preserves_fixed_target_and_limited_scope(study):
    payload = json.loads(json.dumps(_payload(study), allow_nan=False))
    assert len(payload["branches"]) == 2
    for branch in payload["branches"]:
        source = branch["common_source"]
        for node, attributes in source["node_attributes"]:
            history = attributes.get("glyph_history", [])
            if history and history[0] == "deque":
                assert history[1] is None or type(history[1]) is int
                history = history[2]
            assert history == source["state"]["glyph_history"][str(node)]
        assert "z0 is not the actual born EPI pattern" in branch["scope"]
        assert "without replacing that target" in branch["scope"]
        assert "do not prove asymptotic recovery" in branch["scope"]
        for name in ("baseline_flow", "continuation_flow"):
            assert branch[name]["solver_accuracy_certified"] is False
            assert branch[name]["mesh_convergence_certified"] is False
            assert branch[name]["future_or_repeated_behavior_certified"] is False
