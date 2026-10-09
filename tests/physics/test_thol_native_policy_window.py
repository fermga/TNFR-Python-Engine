"""Bounded native-policy continuation and lossless retained-record controls."""

import hashlib
import json
from copy import deepcopy
from fractions import Fraction
from unittest.mock import patch

import pytest

from benchmarks.thol_pressure_feedback import _payload
from tests.physics.test_thol_full_state_response import _metric_readout
from tests.physics.test_thol_native_runtime_response import (  # noqa: F401
    execution as native_execution,
)
from tnfr.research.core_manifests import CoreExperimentManifest

F = Fraction


def _write(path, payload):
    raw = (json.dumps(_payload(payload), allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_pool_roundtrip_is_lossless_for_shared_records_and_reference_like_user_data():
    from benchmarks.thol_native_policy_window import RecordPool, expand_record

    repeated = {
        "epi": (F(1, 3), F(-7, 11)),
        "time": 1.75,
        "inactive_stage": {"outcome": "completed", "before": 0, "after": 0},
    }
    source = {
        "a": repeated,
        "b": repeated,
        "$record": "caller-data",
        "nested": [{"$record": "0" * 64}],
        "false": False,
        "one": 1,
        "signed_zero": ("binary64", (-0.0).hex()),
        "refused": {"outcome": "raised", "error": "gate"},
    }
    pool = RecordPool()
    reference = pool.pack(source)
    expected = _payload(source)
    assert expand_record(reference, pool.nodes) == expected
    size = len(pool.nodes)
    assert pool.pack(deepcopy(source)) == reference and len(pool.nodes) == size
    assert pool.pack(source["a"]) == pool.pack(source["b"])
    repeated["epi"] = (F(99),)
    assert expand_record(reference, pool.nodes) == expected
    restored = expand_record(reference, pool.nodes)
    restored["a"]["epi"][0] = "forged"
    assert expand_record(reference, pool.nodes) == expected
    json.dumps({"record": reference, "pool": pool.nodes}, allow_nan=False)


@pytest.mark.parametrize(
    "reference",
    (
        {"$record": True},
        {"$record": 0},
        {"$record": "0" * 64},
        {"$record": "bad"},
        {"other": "0" * 64},
    ),
)
def test_pool_rejects_invalid_or_absent_references(reference):
    from benchmarks.thol_native_policy_window import expand_record

    with pytest.raises((TypeError, ValueError, KeyError)):
        expand_record(reference, {})


def test_pool_rejects_tampered_content_under_an_unchanged_digest():
    from benchmarks.thol_native_policy_window import RecordPool, expand_record

    pool = RecordPool()
    reference = pool.pack({"x": [1, 2], "scope": "exact record"})
    corrupted = deepcopy(pool.nodes)
    corrupted[reference["$record"]] = {"kind": "forged", "value": 99}
    with pytest.raises((TypeError, ValueError)):
        expand_record(reference, corrupted)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_pool_rejects_nonfinite_scalars_during_encoding_and_decoding(value):
    from benchmarks.thol_native_policy_window import RecordPool, expand_record

    with pytest.raises(ValueError):
        RecordPool().pack(value)
    with pytest.raises(ValueError):
        expand_record(value, {})


def test_pool_budget_exhaustion_produces_no_reference_and_keeps_prior_records_decodable():
    from benchmarks.thol_native_policy_window import RecordPool, expand_record

    pool = RecordPool(max_nodes=1)
    first = pool.pack({"retained": 1})
    with pytest.raises(RuntimeError, match="budget"):
        pool.pack({"later": [2]})
    assert expand_record(first, pool.nodes) == {"retained": 1}


@pytest.fixture(scope="module")
def execution(
    native_execution, tmp_path_factory
):  # noqa: F811 - reused portable fixture
    import benchmarks.thol_native_policy_window as benchmark

    old = native_execution
    native_path = tmp_path_factory.mktemp("policy_window") / "native.json"
    with (
        patch.object(benchmark.native, "run_study", return_value=old["study"]),
        patch(
            "sys.argv",
            [
                "native",
                "--prior-input",
                str(old["path"]),
                "--expected-prior-sha256",
                old["digest"],
                "--output",
                str(native_path),
            ],
        ),
    ):
        benchmark.native.main()
    native_digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    raw_pack = benchmark.RecordPool.pack
    actual_step = benchmark.native.dynamics.step
    packed, calls, admission = [], [], []
    first_by_name = {row["branch"]: row for row in old["study"]["branches"]}

    def record_pack(pool, value):
        original = _payload(value)
        reference = raw_pack(pool, value)
        packed.append((deepcopy(reference), original))
        return reference

    def retained_first(graph, prior):
        first = first_by_name[prior["branch"]]
        assert benchmark.native._record(graph) == first["endpoint"]
        admission.append(prior["branch"])
        return first

    def native_call(graph, **kwargs):
        assert admission == ["control", "child_emission"]
        assert kwargs == {"dt": 0.25, "use_Si": True, "apply_glyphs": True}
        name = next(
            name
            for name, (candidate, _) in old["sources"].items()
            if candidate is graph
        )
        calls.append((name, graph.graph["_t"]))
        return actual_step(graph, **kwargs)

    with (
        patch.object(
            benchmark,
            "replay_response_branch",
            side_effect=lambda name: old["sources"][name],
        ),
        patch.object(
            benchmark.native,
            "run_native_branch",
            retained_first,
        ),
        patch.object(benchmark.native.dynamics, "step", native_call),
        patch.object(benchmark.RecordPool, "pack", record_pack),
    ):
        study = benchmark.run_study(
            native_path,
            old["path"],
            expected_native_sha256=native_digest,
            expected_full_sha256=old["digest"],
        )
    expanded = [
        [benchmark.expand_record(ref, study["record_pool"]) for ref in row["steps"]]
        for row in study["branches"]
    ]
    return {
        "study": study,
        "expanded": expanded,
        "packed": packed,
        "calls": calls,
        "native": old,
        "native_path": native_path,
        "native_digest": native_digest,
        "native_payload": json.loads(native_path.read_bytes()),
    }


def test_exactly_five_additional_calls_preserve_causal_adjacency_and_stop_at_three(
    execution,
):
    study = execution["study"]
    expected_times = [1.75, 2.0, 2.25, 2.5, 2.75]
    assert execution["calls"] == [
        (name, t) for name in ("control", "child_emission") for t in expected_times
    ]
    assert study["protocol"]["maximum_steps_per_branch"] == 5
    assert (
        study["protocol"]["retries"] == 0 and not study["protocol"]["parameter_tuning"]
    )
    for index, (branch, steps) in enumerate(
        zip(study["branches"], execution["expanded"], strict=True)
    ):
        assert branch["status"] == "completed"
        assert branch["attempted_steps"] == branch["completed_steps"] == len(steps) == 5
        assert branch["final_time"] == 3.0
        previous = _payload(execution["native"]["study"]["branches"][index]["endpoint"])
        for ordinal, step in enumerate(steps):
            assert step["ordinal"] == ordinal and step["status"] == "executed"
            assert step["before"] == previous
            assert step["before"]["state"]["time"] == expected_times[ordinal]
            assert step["endpoint"]["state"]["time"] == expected_times[ordinal] + 0.25
            assert step["native_trace"]["native_calls"] == 1
            previous = step["endpoint"]
    assert not study["autonomous_maintenance_certified"]


def test_actual_compaction_roundtrips_every_raw_record_and_original_reference(
    execution,
):
    from benchmarks.thol_native_policy_window import expand_record

    study, pairs = execution["study"], execution["packed"]
    assert pairs
    for reference, original in pairs:
        assert expand_record(reference, study["record_pool"]) == original
    reference = expand_record(study["original_reference"], study["record_pool"])
    assert reference == _payload(
        execution["native"]["study"]["replayed_prior_reports"][0]["original_reference"]
    )
    for retained, first in zip(
        study["first_step_replays"],
        execution["native"]["study"]["branches"],
        strict=True,
    ):
        assert expand_record(retained, study["record_pool"]) == _payload(first)


def test_each_actual_adaptation_gate_retains_counter_inputs_inactive_stages_and_history(
    execution,
):
    for steps in execution["expanded"]:
        for step in steps:
            trace = step["native_trace"]["boundaries"]
            row = next(
                item
                for item in trace
                if item["boundary"] == "adapt_vf_after_structural_stability"
            )
            result = step["adaptation"]
            assert result["available"] and row["outcome"] == "completed"
            cfg = result["configuration"]
            assert cfg["VF_ADAPT_TAU"] == 5
            before, after = row["before"], row["after"]
            old_attrs, new_attrs = dict(before["node_attributes"]), dict(
                after["node_attributes"]
            )
            for i, observed in enumerate(result["nodes"]):
                node = observed["node"]
                si, pressure = before["stored_Si"][i], before["state"]["pressure"][i]
                qualifies = (
                    si >= cfg["si_hi"] and abs(pressure) <= cfg["EPS_DNFR_STABLE"]
                )
                old = old_attrs[node].get("stable_count", 0)
                expected = old + 1 if qualifies else 0
                assert observed["qualifies"] == qualifies
                assert observed["before_count"] == old
                assert (
                    observed["after_count"]
                    == new_attrs[node]["stable_count"]
                    == expected
                )
                assert observed["eligible"] == (expected >= cfg["VF_ADAPT_TAU"])
                if expected < cfg["VF_ADAPT_TAU"]:
                    assert observed["nu_after"] == observed["nu_before"]
            names = [item["boundary"] for item in trace]
            for name in (
                "compute_Si",
                "_apply_glyphs",
                "integrate",
                "coordinate_global_local_phase",
                "_advance_math_engine",
                "_maybe_remesh",
                "_run_validators",
                "_run_after_callbacks",
            ):
                assert names.count(name) == 1
            assert set(step["endpoint_tetrad"]["fields"]) == {
                "phi_s",
                "grad_phi",
                "curv_phi",
                "xi_c",
            }


def _direct_rate(observation, epi):
    snapshot = observation["snapshot"]
    degree, drift = [F(0)] * len(epi), [F(0)] * len(epi)
    for i, j, weight in snapshot["conductance"]:
        weight = F(weight)
        degree[i] += weight
        drift[i] += weight * (epi[j] - epi[i])
    return tuple(
        F(nu) * (F(observation["epi_weight"]) * value / d + F(force))
        for nu, value, d, force in zip(
            snapshot["capacity"], drift, degree, observation["forcing"], strict=True
        )
    )


def test_original_profile_compatibility_is_independent_of_each_evolving_current_profile(
    execution,
):
    original = execution["native"]["study"]["replayed_prior_reports"][0][
        "original_reference"
    ]
    h, z = original["metric_weights"], original["relative_profile"]
    for steps in execution["expanded"]:
        for step in steps:
            for prefix in ("before", "endpoint"):
                observed = step["old_target_" + prefix]
                assert (
                    observed["pattern"]["available"]
                    and observed["compatibility_available"]
                )
                target = observed["target"]
                assert target["target_reference"] == _payload(original)
                epi = tuple(map(F, step[prefix]["state"]["epi"]))
                mean = _metric_readout(epi, h)[0]
                error = tuple(
                    x - mean - profile for x, profile in zip(epi, z, strict=True)
                )
                assert tuple(map(F, target["pattern"]["relative_error"])) == error
                assert (
                    F(target["pattern"]["error_variance"])
                    == sum(w * v * v for w, v in zip(h, error, strict=True)) / 2
                )
                capture = step[prefix + "_capture"]["payload"]["observation"]
                rate = _direct_rate(capture, z)
                rmean = _metric_readout(rate, h)[0]
                residual = tuple(value - rmean for value in rate)
                assert tuple(map(F, target["compatibility_residual"])) == residual
                assert target["target_compatible"] == (not any(residual))
                assert (
                    F(target["compatibility_energy"])
                    == sum(w * v * v for w, v in zip(h, residual, strict=True)) / 2
                )


def test_common_il_formula_retains_generated_and_consumed_pressure_as_different_boundaries(
    execution,
):
    from benchmarks.thol_native_policy_window import expand_record

    study = execution["study"]
    paired = expand_record(study["paired_steps"], study["record_pool"])
    h0 = execution["native"]["study"]["replayed_prior_reports"][0][
        "original_reference"
    ]["metric_weights"]
    assert len(paired) == 5
    assert [row["common_IL_gate"]["available"] for row in paired] == [
        True,
        True,
        False,
        True,
        False,
    ]
    for left, right, result in zip(*execution["expanded"], paired, strict=True):
        assert result["available"]
        for key, boundary in (("before", "before"), ("endpoint", "endpoint")):
            delta = tuple(
                F(b) - F(a)
                for a, b in zip(
                    left[boundary]["state"]["epi"],
                    right[boundary]["state"]["epi"],
                    strict=True,
                )
            )
            mean, centered, full, energy = _metric_readout(delta, h0)
            assert tuple(map(F, result[key]["epi_difference"])) == delta
            assert F(result[key]["weighted_mean_offset"]) == mean
            assert tuple(map(F, result[key]["centered_epi_difference"])) == centered
            assert (
                F(result[key]["centered_H_energy"]) == energy
                and F(result[key]["full_H_energy"]) == full
            )
        gate = result["common_IL_gate"]
        if not gate["available"]:
            assert gate.get("reason")
            continue
        assert all(gate["flags"].values())
        p, q = left["policy_inputs"], right["policy_inputs"]
        generated = p["generation_observation"]
        delta = tuple(map(F, result["before"]["epi_difference"]))
        zero = _direct_rate(generated, (F(0),) * len(delta))
        rate = _direct_rate(generated, delta)
        action = tuple(b - a for a, b in zip(rate, zero, strict=True))
        a = F(gate["retention"])
        ideal = tuple(d - F(1, 4) * a * v for d, v in zip(delta, action, strict=True))
        assert tuple(map(F, gate["generator_action_on_difference"])) == action
        assert tuple(map(F, gate["ideal_difference"])) == ideal
        nu = tuple(map(F, generated["snapshot"]["capacity"]))
        pressure = tuple(
            F(1, 4) * v * a * (F(y) - F(x))
            for v, x, y in zip(
                nu, p["epsilon_pressure"], q["epsilon_pressure"], strict=True
            )
        )
        il = tuple(
            F(1, 4) * v * (F(y) - F(x))
            for v, x, y in zip(nu, p["epsilon_IL"], q["epsilon_IL"], strict=True)
        )
        assert tuple(map(F, gate["pressure_residual_term"])) == pressure
        assert tuple(map(F, gate["IL_residual_term"])) == il
        total = tuple(
            sum(values)
            for values in zip(
                ideal,
                pressure,
                il,
                map(F, gate["integration_residual_difference"]),
                map(F, gate["postintegration_difference"]),
                strict=True,
            )
        )
        assert total == tuple(map(F, result["endpoint"]["epi_difference"]))
        assert not any(map(F, gate["identity_residual"]))


def test_actual_policy_sequence_and_binary64_il_products_are_recorded_not_assumed(
    execution,
):
    for steps in execution["expanded"]:
        policies = []
        for step in steps:
            rows = [
                row
                for row in step["native_trace"]["boundaries"]
                if row["boundary"] == "apply_glyph"
            ]
            assert len(rows) == 16 and all(
                row["outcome"] == "completed" for row in rows
            )
            policy = {row["glyph"] for row in rows}
            assert len(policy) == 1
            policies.append(next(iter(policy)))
            if policy == {"IL"}:
                for index, row in enumerate(rows):
                    exact = F(row["resolved_IL_retention"]) * F(
                        row["before"]["state"]["pressure"][index]
                    )
                    assert F(row["after"]["state"]["pressure"][index]) == F(
                        float(exact)
                    )
            else:
                assert not step["policy_inputs"]["available"]
        assert policies == ["IL", "IL", "EN", "IL", "AL"]


@pytest.mark.parametrize("mutation", ("configuration", "counter", "history"))
def test_full_initial_record_guard_rejects_changes_invisible_to_nodal_state(mutation):
    from benchmarks import thol_native_policy_window as benchmark
    from benchmarks.thol_birth_transport import _birth_graph

    graph = _birth_graph("attached")
    graph.graph["_t"] = 1.75  # Explicit synthetic admission control, not a trajectory.
    first = {"status": "executed", "endpoint": benchmark.native._record(graph)}
    node = next(iter(graph))
    if mutation == "configuration":
        graph.graph["VF_ADAPT_TAU"] = 19
    elif mutation == "counter":
        graph.nodes[node]["stable_count"] = 19
    else:
        graph.nodes[node]["epi_history"] = [19.0]
    assert benchmark.native._state(graph) == first["endpoint"]["state"]
    with pytest.raises(ValueError, match="actual completed"):
        benchmark.continue_window(graph, {}, first, pool=benchmark.RecordPool())


@pytest.mark.parametrize(
    "mutation",
    ("refusal", "start_time", "end_time", "node_order", "old_start", "old_end"),
)
def test_paired_readout_abstains_without_common_completed_times_and_original_space(
    execution, mutation
):
    from benchmarks.thol_native_policy_window import paired_step

    left, right = (deepcopy(rows[0]) for rows in execution["expanded"])
    if mutation == "refusal":
        right["status"] = "refused"
    elif mutation in ("start_time", "end_time"):
        right["before" if mutation == "start_time" else "endpoint"]["state"][
            "time"
        ] += 0.25
    elif mutation == "node_order":
        right["endpoint"]["state"]["nodes"].reverse()
    else:
        # Present domains still agree; the frozen target's domain does not.
        right[
            "old_target_before" if mutation == "old_start" else "old_target_endpoint"
        ]["pattern"]["available"] = False
    metric = execution["native"]["study"]["replayed_prior_reports"][0][
        "original_reference"
    ]["metric_weights"]
    result = paired_step(left, right, metric)
    assert not result["available"] and result["reason"]


@pytest.mark.parametrize("field", ("forcing", "retention", "capacity"))
def test_paired_il_identity_abstains_when_generation_models_differ(execution, field):
    from benchmarks.thol_native_policy_window import paired_step

    left, right = (deepcopy(rows[0]) for rows in execution["expanded"])
    inputs = right["policy_inputs"]
    if field == "retention":
        inputs[field] = "1/2"
    else:
        owner = inputs["generation_observation"]
        if field == "capacity":
            owner = owner["snapshot"]
        owner[field][0] = str(F(owner[field][0]) + 1)
    metric = execution["native"]["study"]["replayed_prior_reports"][0][
        "original_reference"
    ]["metric_weights"]
    result = paired_step(left, right, metric)
    assert result["available"] and not result["common_IL_gate"]["available"]
    assert not result["common_IL_gate"]["flags"][field]


def _records(payload):
    yield payload["before"]
    yield payload["endpoint"]
    for row in payload["native_trace"]["boundaries"]:
        yield row["before"]
        yield row["after"]


def test_native_replay_allows_only_consistent_child_utc_in_every_enumerated_record(
    execution,
):
    from benchmarks.thol_native_policy_window import _admit_native_replay

    first = execution["native"]["study"]["branches"][1]
    old = _payload(first)
    children = execution["native"]["study"]["replayed_prior_reports"][1]["lineage"][
        "children"
    ]
    for record in _records(old):
        attrs = dict(record["node_attributes"])[children[0]]
        attrs["emission_timestamp"] = attrs["_emission_origin"] = (
            "2031-01-01T00:00:00+00:00"
        )
        attrs["_structural_lineage"]["origin"] = attrs["emission_timestamp"]
    before = deepcopy(old)
    result = _admit_native_replay(first, old, children)
    assert result["scientific_payload_equal"] and not result["full_payload_equal"]
    assert old == before


@pytest.mark.parametrize(
    "mutation",
    (
        "utc_alias",
        "physical_time",
        "activation",
        "history",
        "nonchild_utc",
        "held_prediction",
    ),
)
def test_native_replay_rejects_scientific_or_unlisted_metadata_changes(
    execution, mutation
):
    from benchmarks.thol_native_policy_window import _admit_native_replay

    first = execution["native"]["study"]["branches"][1]
    old = _payload(first)
    children = execution["native"]["study"]["replayed_prior_reports"][1]["lineage"][
        "children"
    ]
    attrs = dict(old["endpoint"]["node_attributes"])
    child = attrs[children[0]]
    if mutation == "utc_alias":
        child["_emission_origin"] = "2031-01-01T00:00:00+00:00"
    elif mutation == "physical_time":
        old["endpoint"]["state"]["time"] += 0.25
    elif mutation == "activation":
        child["activation_count"] = 999
    elif mutation == "history":
        child["epi_time_history"] = ["forged"]
    elif mutation == "nonchild_utc":
        parent = next(node for node in attrs if node not in children)
        attrs[parent]["emission_timestamp"] = "2031-01-01T00:00:00+00:00"
    else:
        old["held_prediction"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _admit_native_replay(first, old, children)


def test_refusal_stops_window_after_one_attempt_and_keeps_partial_record(monkeypatch):
    from benchmarks import thol_native_policy_window as benchmark
    from benchmarks.thol_birth_transport import _birth_graph

    graph = _birth_graph("attached")
    graph.graph["_t"] = 1.75  # Synthetic stop-control; no physical replay claim.
    capture = benchmark.native._capture(graph)
    assert capture["available"]
    prior = {
        "branch": "control",
        "prefix": {"coupling": {"refreshed_forcing": capture["payload"]}},
    }
    first = {"status": "executed", "endpoint": benchmark.native._record(graph)}
    calls = []

    def refusal(graph, **kwargs):
        calls.append(kwargs)
        graph.graph["synthetic_partial_marker"] = True
        return {
            "status": "refused",
            "boundaries": [],
            "captures": {},
            "native_calls": 1,
            "failure": {
                "type": "OperatorPreconditionError",
                "message": "bounded control",
            },
        }

    monkeypatch.setattr(benchmark.native, "_trace_step", refusal)
    monkeypatch.setattr(
        benchmark.native,
        "_capture",
        lambda graph: {"available": False, "reason": "synthetic"},
    )
    pool = benchmark.RecordPool()
    result = benchmark.continue_window(graph, prior, first, pool=pool)
    assert (
        len(calls) == result["attempted_steps"] == 1 and result["completed_steps"] == 0
    )
    assert result["status"] == "refused" and result["final_time"] == 1.75
    step = benchmark.expand_record(result["steps"][0], pool.nodes)
    assert step["native_trace"]["failure"]["message"] == "bounded control"
    assert step["endpoint"]["graph_attributes"]["synthetic_partial_marker"] is True
    assert not step["policy_inputs"]["available"]
    assert "synthetic_partial_marker" not in first["endpoint"]["graph_attributes"]


@pytest.mark.parametrize(
    "mutation", ("claim", "ancestry_digest", "ancestry_manifest", "order", "refusal")
)
def test_loader_rejects_incorrect_claim_ancestry_or_incomplete_branches_with_fresh_digest(
    execution, tmp_path, mutation
):
    from benchmarks.thol_native_policy_window import load_evidence

    payload = deepcopy(execution["native_payload"])
    if mutation == "claim":
        payload["manifest"]["claim_id"] = "unrelated"
    elif mutation == "ancestry_digest":
        payload["prior_evidence"]["sha256"] = "0" * 64
    elif mutation == "ancestry_manifest":
        payload["prior_evidence"]["historical_manifest"]["seed"] = 9
    elif mutation == "order":
        payload["branches"].reverse()
    else:
        payload["branches"][1]["status"] = "refused"
    path = tmp_path / "forged.json"
    digest = _write(path, payload)
    with pytest.raises(ValueError):
        load_evidence(
            path,
            execution["native"]["path"],
            expected_native_sha256=digest,
            expected_full_sha256=execution["native"]["digest"],
        )


def test_cli_keeps_lossless_evidence_and_rejects_overwriting_either_input(
    execution, tmp_path, monkeypatch
):
    import benchmarks.thol_native_policy_window as benchmark

    output = tmp_path / "window.json"
    monkeypatch.setattr(
        benchmark, "run_study", lambda *args, **kwargs: execution["study"]
    )
    args = [
        "window",
        "--native-input",
        str(execution["native_path"]),
        "--full-response-input",
        str(execution["native"]["path"]),
    ]
    monkeypatch.setattr("sys.argv", args + ["--output", str(output)])
    benchmark.main()
    result = json.loads(output.read_bytes())
    CoreExperimentManifest(**result["manifest"]).validate_for_admission()
    assert result["experimental_status"] == "No empirical correspondence tested"
    assert result["record_pool"] == _payload(execution["study"]["record_pool"])
    for path in (execution["native_path"], execution["native"]["path"]):
        before = path.read_bytes()
        monkeypatch.setattr("sys.argv", args + ["--output", str(path)])
        with pytest.raises(ValueError, match="must not overwrite"):
            benchmark.main()
        assert path.read_bytes() == before
