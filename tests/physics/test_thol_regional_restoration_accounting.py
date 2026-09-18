"""Portable accounting controls built from explicit synthetic boundary records.

These records test exact arithmetic and decision semantics, not execution
provenance. No historical study, pressure kernel or runtime producer is run.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
import json

import pytest

from benchmarks import thol_regional_restoration_accounting as audit
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import NonEpiForcingObservation
from tnfr.physics.support_transport import _from_data


NODES = (0, 1, 2, 3)
CHILDREN = (2, 3)
CAPACITY = (F(1), F(2), F(1), F(2))
UNDIRECTED = ((0, 1), (1, 2), (2, 3), (0, 3))
EDGES = tuple(sorted((i, j, F(1)) for a, b in UNDIRECTED for i, j in ((a, b), (b, a))))
SUPPORT = tuple(tuple(j for i, j, _ in EDGES if i == k) for k in NODES)
ZERO = (F(0),)*4
EPI_WEIGHT = F(1, 2)
DT = F(1, 4)


def _add(left, right):
    return tuple(a+b for a, b in zip(left, right, strict=True))


def _sub(left, right):
    return tuple(a-b for a, b in zip(left, right, strict=True))


def _vector(index, value):
    return tuple(F(value) if i == index else F(0) for i in NODES)


def _capture(epi, pressure, phase, forcing, kernel=ZERO):
    snapshot = _from_data(NODES, EDGES, SUPPORT, epi, CAPACITY, pressure)
    model = tuple(EPI_WEIGHT*g+f for g, f in zip(snapshot.epi_gradient, forcing, strict=True))
    fresh = _add(model, kernel)
    observation = NonEpiForcingObservation(
        snapshot=snapshot, phase=phase, epi_weight=EPI_WEIGHT, forcing=forcing,
        phase_gradient=tuple(2*f for f in forcing),
        normalized_weights=(("phase", F(1, 2)), ("epi", F(1, 2)), ("vf", F(0)), ("topo", F(0))),
        full_kernel_pressure=fresh, kernel_pressure_defect=kernel,
        stored_pressure_residual=_sub(pressure, fresh))
    return {"available": True, "payload": {"observation": asdict(observation),
            "components": (("phase", forcing), ("vf", ZERO), ("topo", ZERO))}}


def _record(epi, pressure, phase, time):
    return {"state": {"nodes": list(NODES), "epi": list(map(float, epi)),
                      "capacity": list(map(float, CAPACITY)), "pressure": list(map(float, pressure)),
                      "phase": list(map(float, phase)), "time": float(time),
                      "edges": [(i, j, {"weight": 1.0}) for i, j in UNDIRECTED]},
            "ordered_neighbors": [(i, list(SUPPORT[i])) for i in NODES],
            "node_attributes": [], "graph_attributes": {}}


def _fixture(*, control_scale=F(1), error_scale=F(1, 4), lags=False, zero_control=False, zero_error=False):
    """Six chosen finite endpoint pairs with exact stage/pressure bookkeeping."""
    control_initial = (F(1, 4), F(1, 2), F(3, 4), F(1, 4))
    if zero_control:
        control_initial = (F(1, 4), F(1, 2), F(1, 2), F(1, 2))
    initial_jump = F(0) if zero_error else F(1, 4)
    perturbed_initial = _add(control_initial, _vector(2, initial_jump))
    reference_snapshot = _from_data(NODES, EDGES, SUPPORT, control_initial, CAPACITY, ZERO)
    original = derive_forced_support_balance(reference_snapshot, epi_weight=EPI_WEIGHT, forcing=ZERO)
    branches = []
    for branch in (0, 1):
        records = []
        current = control_initial if branch == 0 else perturbed_initial
        previous = _record(current, ZERO, ZERO, F(3, 2))
        for ordinal in range(6):
            time = F(3, 2)+ordinal*DT
            source = _vector(2, F(1, 32)) if lags and branch else ZERO
            kernel = _vector(2, F(1, 512)) if lags and branch else ZERO
            stored = _vector(3, F(-1, 1024)) if lags and branch else ZERO
            geometric = _from_data(NODES, EDGES, SUPPORT, current, CAPACITY, ZERO)
            generated = _add(tuple(EPI_WEIGHT*g for g in geometric.epi_gradient), _add(source, _add(kernel, stored)))
            phase0 = tuple(F(value) for value in previous["state"]["phase"])
            gen_record = _record(current, generated, phase0, time)
            reset_epi = _add(current, _vector(2, F(1, 64))) if lags and branch else current
            reset_record = _record(reset_epi, generated, phase0, time)
            entry_pressure = _add(generated, _vector(3, F(1, 128))) if lags and branch else generated
            entry_source = _add(source, _vector(3, F(1, 256))) if lags and branch else source
            entry_phase = _add(phase0, _vector(3, F(1, 8))) if lags and branch else phase0
            entry = _record(reset_epi, entry_pressure, entry_phase, time)
            scale = 1+F(ordinal+1, 8)*(control_scale-1)
            next_control = tuple((F(1, 2)+scale*(value-F(1, 2))) if i in CHILDREN else value
                                 for i, value in enumerate(control_initial))
            amplitude = initial_jump*(1+F(ordinal+1, 8)*(error_scale-1))
            final_epi = _add(next_control, _vector(2, amplitude)) if branch else next_control
            post_jump = _vector(2, F(1, 2048)) if lags and branch else ZERO
            integrated = _record(_sub(final_epi, post_jump), entry_pressure, entry_phase, time+DT)
            endpoint = _record(final_epi, entry_pressure, entry_phase, time+DT)
            rows = [
                {"ordinal": 0, "boundary": "_prepare_dnfr", "outcome": "completed", "before": previous, "after": gen_record},
                {"ordinal": 1, "boundary": "apply_glyph", "glyph": "EN", "node": 2, "outcome": "completed",
                 "before": gen_record, "after": reset_record},
                {"ordinal": 2, "boundary": "apply_glyph", "glyph": "IL", "node": 3, "outcome": "completed",
                 "before": reset_record, "after": entry},
                {"ordinal": 3, "boundary": "integrate", "outcome": "completed", "before": entry, "after": integrated},
            ]
            trace = {"status": "executed", "boundaries": rows, "captures": {
                "pressure_generation": _capture(current, generated, phase0, source, kernel),
                "integrator_entry": _capture(reset_epi, entry_pressure, entry_phase, entry_source)}}
            records.append({"ordinal": ordinal, "status": "executed", "before": previous,
                            "endpoint": endpoint, "native_trace": trace})
            previous, current = endpoint, final_epi
        branches.append(records)
    return (*branches, original, CHILDREN, control_initial, perturbed_initial)


def _energy(values, metric):
    mean = sum(h*x for h, x in zip(metric, values, strict=True))/sum(metric)
    return sum(h*(x-mean)**2/2 for h, x in zip(metric, values, strict=True))


def _center(values, metric):
    mean = sum(h*x for h, x in zip(metric, values, strict=True))/sum(metric)
    return tuple(x-mean for x in values)


@pytest.mark.parametrize("lags", (False, True))
def test_complete_exact_endpoint_and_stage_energy_accounting(lags):
    arguments = _fixture(lags=lags)
    original = arguments[2]
    result = audit.analyze_response(*arguments)
    weights = tuple(original.metric_weights[i] for i in CHILDREN)
    assert result["completed_pairs"] == 6 and len(result["endpoints"]) == 7
    assert tuple(row["time"] for row in result["endpoints"]) == tuple(F(6+k, 4) for k in range(7))
    assert result["decision"]["outcome"] == "supported_in_scope"
    assert not result["autonomous_maintenance_certified"]
    assert not result["decision"]["autonomous_maintenance_certified"]
    for row in result["endpoints"]:
        difference = _sub(row["perturbed_epi"], row["control_epi"])
        expected = _energy(difference, weights)
        assert row["paired"]["centered_H_energy"] == expected
        assert row["control_variance"] == _energy(row["control_epi"], weights)
        assert row["relative_error"]["value"] == expected/row["control_variance"]
    accumulated = F(0)
    for step in result["steps"]:
        increment = F(0)
        for stage in step["budgets"].values():
            before = tuple(stage["before"]["epi"][i] for i in CHILDREN)
            after = tuple(stage["after"]["epi"][i] for i in CHILDREN)
            expected = _energy(after, weights)-_energy(before, weights)
            assert stage["variance_change"] == expected
            assert stage["variance_identity_residual"] == stage["mass_identity_residual"] == 0
            increment += expected
        assert increment == step["variance_change"]
        assert step["local_glyphs_available"]
        assert sum(row["budget"]["variance_change"] for row in step["local_glyphs"])+sum(
            row["variance_change"] for row in step["intervening_epi_budgets"]
        ) == step["budgets"]["pre_integration"]["variance_change"]
        accumulated += increment
    assert result["variance_change"] == accumulated
    assert accumulated == result["endpoints"][-1]["paired"]["centered_H_energy"]-result["endpoints"][0]["paired"]["centered_H_energy"]
    assert result["variance_telescope_residual"] == 0
    json.dumps(audit._payload(result), allow_nan=False)


def test_all_five_pressure_terms_have_independent_sign_and_work_checks():
    arguments = _fixture(lags=True)
    result = audit.analyze_response(*arguments)
    first = result["steps"][0]
    parts = first["pressure_parts"]
    assert parts["generation_kernel_defect"]["pressure"] == _vector(2, F(1, 512))
    assert parts["generation_stored_defect"]["pressure"] == _vector(3, F(-1, 1024))
    assert parts["pressure_operator_write"]["pressure"] == _vector(3, F(1, 128))
    assert parts["held_source_lag"]["pressure"] == _vector(3, F(-1, 256))
    jump = _vector(2, F(1, 64))
    negative_laplacian = tuple((jump[(i-1) % 4]+jump[(i+1) % 4])/2-jump[i] for i in NODES)
    assert parts["held_epi_lag"]["pressure"] == tuple(-EPI_WEIGHT*x for x in negative_laplacian)
    integration = first["budgets"]["integration"]
    full = tuple(sum(part["pressure"][i] for part in parts.values()) for i in NODES)
    assert full == integration["balance"]["stored_pressure_defect"]
    weights = tuple(arguments[2].metric_weights[i] for i in CHILDREN)
    centered = _center(tuple(integration["before"]["epi"][i] for i in CHILDREN), weights)
    for part in parts.values():
        expected_mass = DT*sum(2*part["pressure"][i] for i in CHILDREN)
        expected_shape = DT*sum(2*z*part["pressure"][i] for i, z in zip(CHILDREN, centered, strict=True))
        assert part["weighted_total_work"] == expected_mass
        assert part["variance_work"] == expected_shape
    assert sum(part["variance_work"] for part in parts.values()) == DT*integration["balance"]["variance_defect_rate"]
    assert first["source_channels"]["phase"]["pressure"] == _add(_vector(2, F(1, 32)), _vector(3, F(1, 256)))
    assert first["source_channels"]["vf"]["pressure"] == first["source_channels"]["topo"]["pressure"] == ZERO
    assert any(integration["state_defect"])


def test_boundary_and_local_pressure_work_remain_separate_exact_terms():
    result = audit.analyze_response(*_fixture(lags=True))
    for step in result["steps"]:
        integration = step["budgets"]["integration"]
        delta = integration["before"]["epi"]
        mean = (2*delta[2]+delta[3])/3
        # Full-graph cut edges are 2 -> 1 and 3 -> 0, both unit weight.
        expected_self = -EPI_WEIGHT*((delta[2]-mean)**2+(delta[3]-mean)**2)
        expected_input = EPI_WEIGHT*(
            (delta[2]-mean)*(delta[1]-mean) + (delta[3]-mean)*(delta[0]-mean))
        split = step["boundary_split"]
        assert split["self_relaxation_rate"] == expected_self < 0
        assert split["incoming_field_rate"] == expected_input
        assert expected_self+expected_input == integration["balance"]["variance_boundary_rate"]
        assert split["self_relaxation_work"] == DT*expected_self
        assert split["incoming_field_work"] == DT*expected_input
        local = step["local_glyphs"]
        assert local[0]["pressure_write"]["pressure"] == ZERO
        assert local[1]["pressure_write"]["pressure"] == _vector(3, F(1, 128))
        assert step["unassigned_pressure_write"]["pressure"] == ZERO
        for field in ("variance_work", "weighted_total_work"):
            assigned = sum(row["pressure_write"][field] for row in local)
            assert assigned+step["unassigned_pressure_write"][field] == step["pressure_parts"]["pressure_operator_write"][field]
    assert not result["decision"]["autonomous_maintenance_certified"]


def test_decreased_raw_and_relative_error_with_control_flattening_is_rejected():
    result = audit.analyze_response(*_fixture(control_scale=F(1, 2)))
    decision = result["decision"]
    assert decision["gates"]["raw_error_decreased"]
    assert decision["gates"]["relative_error_decreased"]
    assert not decision["gates"]["control_contrast_not_attenuated"]
    assert decision["outcome"] == "rejected_in_scope"
    assert not decision["original_control_form_unchanged"]


def _score(error, contrast, drift=0):
    return {"paired": {"centered_H_energy": F(error)}, "control_variance": F(contrast),
            "control_centered_drift_energy": F(drift)}


@pytest.mark.parametrize("scores,complete,outcome,failed", [
    ([_score(4, 2), _score(1, 2)], True, "supported_in_scope", None),
    ([_score(4, 2), _score(1, 2)], False, "inconclusive", "complete_fixed_horizon"),
    ([_score(0, 2), _score(0, 2)], True, "inconclusive", "initial_nonuniform_damage"),
    ([_score(4, 2), _score(1, 0), _score(1, 2)], True, "inconclusive", "control_contrast_positive_everywhere"),
    ([_score(4, 2), _score(5, 4)], True, "rejected_in_scope", "raw_error_decreased"),
    ([_score(4, 2), _score(3, 1)], True, "rejected_in_scope", "relative_error_decreased"),
    ([_score(4, 2), _score(1, 1)], True, "rejected_in_scope", "control_contrast_not_attenuated"),
])
def test_decision_keeps_failed_gates_separate(scores, complete, outcome, failed):
    result = audit.classify_response(scores, complete=complete)
    assert result["outcome"] == outcome
    if failed is not None:
        assert failed in result["failed_gates"]
    assert not result["autonomous_maintenance_certified"]


def test_supported_tracking_need_not_preserve_original_control_form():
    result = audit.classify_response([_score(4, 2), _score(1, 3, 9)], complete=True)
    assert result["outcome"] == "supported_in_scope"
    assert not result["original_control_form_unchanged"]


@pytest.mark.parametrize("zero_control,zero_error", [(True, False), (False, True)])
def test_singular_control_or_absent_damage_is_inconclusive(zero_control, zero_error):
    result = audit.analyze_response(*_fixture(zero_control=zero_control, zero_error=zero_error))
    assert result["decision"]["outcome"] == "inconclusive"
    if zero_control:
        assert all(not row["relative_error"]["available"] for row in result["endpoints"])


def test_refusal_scores_only_completed_aligned_prefix():
    arguments = _fixture()
    arguments[1][2]["native_trace"]["status"] = "refused"
    result = audit.analyze_response(*arguments)
    assert result["completed_pairs"] == 2 and len(result["endpoints"]) == 3
    assert result["decision"]["outcome"] == "inconclusive"


@pytest.mark.parametrize("field", ("capacity", "nodes", "edges", "outer_time", "entry_time", "exit_time", "generation_time"))
def test_changed_primitive_domain_or_clock_is_rejected(field):
    arguments = _fixture()
    step = arguments[1][0]
    if field == "capacity":
        step["endpoint"]["state"]["capacity"][0] = 3.0
    elif field == "nodes":
        step["endpoint"]["state"]["nodes"].reverse()
    elif field == "edges":
        step["endpoint"]["state"]["edges"].pop()
    elif field == "outer_time":
        step["endpoint"]["state"]["time"] = 2.0
    elif field == "entry_time":
        step["native_trace"]["boundaries"][-1]["before"]["state"]["time"] = 2.0
    elif field == "exit_time":
        step["native_trace"]["boundaries"][-1]["after"]["state"]["time"] = 2.0
    else:
        step["native_trace"]["boundaries"][0]["after"]["state"]["time"] = 2.0
    with pytest.raises(ValueError):
        audit.analyze_response(*arguments)


def test_nonchronological_generation_and_integration_are_rejected():
    arguments = _fixture()
    arguments[1][0]["native_trace"]["boundaries"].reverse()
    with pytest.raises(ValueError):
        audit.analyze_response(*arguments)


@pytest.mark.parametrize("position", ("before_generation", "after_integration"))
def test_glyph_receipt_outside_held_pressure_interval_is_rejected(position):
    arguments = _fixture(lags=True)
    rows = arguments[1][0]["native_trace"]["boundaries"]
    glyph = rows.pop(1)
    rows.insert(0 if position == "before_generation" else len(rows), glyph)
    with pytest.raises(ValueError, match="glyph.*generation|glyph.*integration"):
        audit.analyze_response(*arguments)


@pytest.mark.parametrize("fault", ("missing_capture", "wrong_capture", "metric", "adjacent_record", "unmatched_prefix", "too_many"))
def test_incomplete_or_forged_accounting_inputs_reject(fault):
    arguments = list(_fixture())
    if fault == "missing_capture":
        arguments[1][0]["native_trace"]["captures"]["integrator_entry"]["available"] = False
    elif fault == "wrong_capture":
        arguments[1][0]["native_trace"]["captures"]["integrator_entry"]["payload"]["observation"]["kernel_pressure_defect"] = _vector(2, F(1))
    elif fault == "metric":
        arguments[2] = replace(arguments[2], metric_weights=(F(1),)*4)
    elif fault == "adjacent_record":
        arguments[1][1]["before"] = deepcopy(arguments[1][1]["before"])
        arguments[1][1]["before"]["graph_attributes"]["modified"] = True
    elif fault == "unmatched_prefix":
        arguments[1].pop()
    else:
        arguments[0].append(deepcopy(arguments[0][-1]))
        arguments[1].append(deepcopy(arguments[1][-1]))
    with pytest.raises(ValueError):
        audit.analyze_response(*arguments)


def test_inputs_unchanged_and_no_execution_owner_called(monkeypatch):
    arguments = _fixture(lags=True)
    before = deepcopy(arguments)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure accounting must not execute a producer or kernel")

    from tnfr import dynamics
    from tnfr.dynamics import coordination
    from tnfr.physics import forcing_realization
    monkeypatch.setattr(dynamics, "step", forbidden)
    monkeypatch.setattr(coordination, "coordinate_global_local_phase", forbidden)
    monkeypatch.setattr(forcing_realization, "capture_non_epi_forcing", forbidden)
    audit.analyze_response(*arguments)
    assert arguments == before
