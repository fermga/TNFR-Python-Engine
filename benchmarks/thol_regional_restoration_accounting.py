"""Pure finite regional response accounting for a declared paired native run.

This reader does not execute operators or infer pressure from endpoint motion.
Its public records are data; the invoking producer owns execution provenance.
"""

from dataclasses import asdict
from fractions import Fraction as F

from benchmarks.thol_family_closure import _capture, _equal, _match_live_state
from benchmarks.thol_full_state_response import _paired_delta, _payload
from benchmarks.thol_regional_balance_audit import _bind_source_edges, _component_budget
from benchmarks.thol_regional_recovery_audit import _ratio
from tnfr.physics.support_transport import _from_data, observe_regional_support_euler

START, DT, COUNT = F(3, 2), F(1, 4), 6


def _v(values):
    return tuple(F(x) for x in values)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b, strict=True))


def _sum(vectors, size):
    return tuple(sum((v[i] for v in vectors), F(0)) for i in range(size))


def _one(trace, name):
    rows = [row for row in trace["boundaries"] if row["boundary"] == name]
    if len(rows) != 1 or rows[0]["outcome"] != "completed":
        raise ValueError(f"one completed {name} boundary required")
    return rows[0]


def _domain(record, reference):
    state, source = record["state"], reference.source
    if tuple(state["nodes"]) != source.nodes or _v(state["capacity"]) != source.capacity:
        raise ValueError("fixed ordered nodes and capacity required")
    _bind_source_edges(state, source)
    if len(state["epi"]) != len(source.nodes):
        raise ValueError("complete EPI vector required")
    return _v(state["epi"])


def _captured(trace, key, record, reference):
    raw = trace["captures"][key]
    if not raw["available"]:
        raise ValueError(f"{key} capture unavailable")
    snap, obs, components = _capture(raw["payload"])
    _match_live_state(record["state"], raw["payload"], snap)
    _domain(record, reference)
    for name in ("nodes", "conductance", "support_neighbors", "capacity"):
        if getattr(snap, name) != getattr(reference.source, name):
            raise ValueError("capture differs from fixed full domain")
    if obs.epi_weight != reference.epi_weight:
        raise ValueError("captured EPI coefficient changed")
    return snap, obs, dict(components)


def endpoint_readout(control, perturbed, reference, children, initial_control):
    """Separate paired shape, mean, control contrast and control-form drift."""
    indices = tuple(reference.source.nodes.index(node) for node in children)
    metric = tuple(reference.metric_weights[i] for i in indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(w <= 0 for w in metric) or len(control) != len(reference.source.nodes)
            or len(perturbed) != len(control) or len(initial_control) != len(control)):
        raise ValueError("complete vectors and a distinct positive-metric region required")
    x, y, x0 = (tuple(_v(values)[i] for i in indices)
                for values in (control, perturbed, initial_control))
    paired = _paired_delta(x, y, metric)
    shape = _paired_delta((F(0),)*len(indices), x, metric)
    drift = _paired_delta(x0, x, metric)
    parents = tuple(i for i in range(len(control)) if i not in indices)
    full_delta = _sub(_v(perturbed), _v(control))
    parent_mean = (sum((reference.metric_weights[i]*full_delta[i] for i in parents), F(0))
                   / sum(reference.metric_weights[i] for i in parents)) if parents else None
    return {
        "control_epi": x, "perturbed_epi": y, "paired": paired,
        "control_mean": shape["weighted_mean_offset"],
        "control_centered_epi": shape["centered_epi_difference"],
        "control_variance": shape["centered_H_energy"],
        "control_centered_drift": drift["centered_epi_difference"],
        "control_centered_drift_energy": drift["centered_H_energy"],
        "parent_mean_offset": parent_mean,
        "child_parent_mean_contrast": paired["weighted_mean_offset"]-parent_mean
        if parent_mean is not None else None,
        "relative_error": _ratio(paired["centered_H_energy"], shape["centered_H_energy"]),
    }


def classify_response(endpoints, *, complete):
    """Predeclared sufficient finite tracking criterion, not an autonomy test."""
    if not endpoints:
        raise ValueError("initial endpoint required")
    first, last = endpoints[0], endpoints[-1]
    e0, e1 = (row["paired"]["centered_H_energy"] for row in (first, last))
    v0, v1 = (row["control_variance"] for row in (first, last))
    positive = all(row["control_variance"] > 0 for row in endpoints)
    gates = {
        "complete_fixed_horizon": complete,
        "initial_nonuniform_damage": e0 > 0,
        "control_contrast_positive_everywhere": positive,
        "raw_error_decreased": e1 < e0,
        "relative_error_decreased": e1*v0 < e0*v1 if positive else False,
        "control_contrast_not_attenuated": v1 >= v0,
    }
    if not complete or not gates["initial_nonuniform_damage"] or not positive:
        outcome = "inconclusive"
    else:
        outcome = "supported_in_scope" if all(gates.values()) else "rejected_in_scope"
    return {
        "outcome": outcome, "gates": gates,
        "failed_gates": tuple(key for key, value in gates.items() if not value),
        "original_control_form_unchanged": all(row["control_centered_drift_energy"] == 0 for row in endpoints),
        "autonomous_maintenance_certified": False,
        "scope": "Finite configured convergence toward the independently evolving control only. "
                 "Nonattenuated contrast is a conservative sufficient protocol gate, not a necessary identity law. "
                 "Passive transport, source forcing and supplied scheduling remain separate mechanisms.",
    }


def analyze_response(control_steps, perturbed_steps, original_reference, children,
                     initial_control_epi, initial_perturbed_epi):
    """Account for the completed aligned prefix, at most six quarter-time steps."""
    reference, source = original_reference, original_reference.source
    size = len(source.nodes)
    if any(value <= 0 for value in source.capacity):
        raise ValueError("positive full capacity required")
    strengths = tuple(sum((w for i, _j, w in source.conductance if i == k), F(0)) for k in range(size))
    if any(d <= 0 for d in strengths):
        raise ValueError("positive full strength required")
    _equal(tuple(d/nu for d, nu in zip(strengths, source.capacity, strict=True)),
           reference.metric_weights, "full fixed metric")
    if len(control_steps) != len(perturbed_steps) or len(control_steps) > COUNT:
        raise ValueError("aligned prefix of at most six steps required")
    children = tuple(children)
    initial_control_epi, initial_perturbed_epi = _v(initial_control_epi), _v(initial_perturbed_epi)
    current = (initial_control_epi, initial_perturbed_epi)
    endpoints = [{"time": START, **endpoint_readout(*current, reference, children, initial_control_epi)}]
    budgets = []
    previous_records = None
    for ordinal, pair in enumerate(zip(control_steps, perturbed_steps, strict=True)):
        pair = tuple(_payload(row) for row in pair)
        if any(step["native_trace"]["status"] != "executed" for step in pair):
            break
        expected_time = START+ordinal*DT
        for index, step in enumerate(pair):
            for key, time in (("before", expected_time), ("endpoint", expected_time+DT)):
                _domain(step[key], reference)
                if F(step[key]["state"]["time"]) != time:
                    raise ValueError("predeclared native clock differs")
            if _v(step["before"]["state"]["epi"]) != current[index]:
                raise ValueError("adjacent EPI records differ")
            if previous_records is not None:
                _equal(previous_records[index], step["before"], "complete adjacent native record")
        traces = tuple(step["native_trace"] for step in pair)
        generation_rows = tuple(_one(t, "_prepare_dnfr") for t in traces)
        integration_rows = tuple(_one(t, "integrate") for t in traces)
        for trace, generation_row, integration_row in zip(traces, generation_rows, integration_rows, strict=True):
            gen_position = trace["boundaries"].index(generation_row)
            integration_position = trace["boundaries"].index(integration_row)
            if gen_position >= integration_position:
                raise ValueError("pressure generation must precede integration")
            if any(not gen_position < i < integration_position for i, row in enumerate(trace["boundaries"])
                   if row["boundary"] == "apply_glyph"):
                raise ValueError("named glyph writes must lie between generation and integration")
            for row, key, time in ((generation_row, "before", expected_time),
                                   (generation_row, "after", expected_time),
                                   (integration_row, "before", expected_time),
                                   (integration_row, "after", expected_time+DT)):
                _domain(row[key], reference)
                if F(row[key]["state"]["time"]) != time:
                    raise ValueError("internal generation/integration clock differs")
        generation = tuple(_captured(t, "pressure_generation", row["after"], reference)
                           for t, row in zip(traces, generation_rows, strict=True))
        entry = tuple(_captured(t, "integrator_entry", row["before"], reference)
                      for t, row in zip(traces, integration_rows, strict=True))
        pressure = _sub(entry[1][0].stored_pressure, entry[0][0].stored_pressure)
        forcing = _sub(entry[1][1].forcing, entry[0][1].forcing)

        def snapshot(epi):
            return _from_data(source.nodes, source.conductance, source.support_neighbors,
                              epi, source.capacity, pressure)

        def budget(before, after, dt=F(0)):
            return observe_regional_support_euler(snapshot(before), snapshot(after), children,
                                                  dt=dt, epi_weight=reference.epi_weight, forcing=forcing)

        before_delta = _sub(current[1], current[0])
        gen_delta = _sub(generation[1][0].epi, generation[0][0].epi)
        entry_delta = _sub(entry[1][0].epi, entry[0][0].epi)
        for row in integration_rows:
            _domain(row["after"], reference)
            if _v(row["after"]["state"]["pressure"]) != _v(row["before"]["state"]["pressure"]):
                raise ValueError("integrator did not retain the captured pressure")
        exit_delta = _sub(*(_v(row["after"]["state"]["epi"]) for row in reversed(integration_rows)))
        final = tuple(_v(step["endpoint"]["state"]["epi"]) for step in pair)
        final_delta = _sub(final[1], final[0])
        staged = {"pre_generation": budget(before_delta, gen_delta),
                  "pre_integration": budget(gen_delta, entry_delta),
                  "integration": budget(entry_delta, exit_delta, DT),
                  "post_integration": budget(exit_delta, final_delta)}
        integration = staged["integration"]
        gen_pressure = _sub(generation[1][0].stored_pressure, generation[0][0].stored_pressure)
        gen_force = _sub(generation[1][1].forcing, generation[0][1].forcing)
        gen_gradient = snapshot(gen_delta).epi_gradient
        pressure_parts = {
            "generation_kernel_defect": _sub(generation[1][1].kernel_pressure_defect, generation[0][1].kernel_pressure_defect),
            "generation_stored_defect": _sub(generation[1][1].stored_pressure_residual, generation[0][1].stored_pressure_residual),
            "pressure_operator_write": _sub(pressure, gen_pressure),
            "held_epi_lag": tuple(reference.epi_weight*v for v in _sub(gen_gradient, integration.before.epi_gradient)),
            "held_source_lag": _sub(gen_force, forcing),
        }
        _equal(_sum(tuple(pressure_parts.values()), size), integration.balance.stored_pressure_defect,
               "generation, operator and held-input pressure split")
        channels = {key: _sub(entry[1][2][key], values) for key, values in entry[0][2].items()}
        _equal(_sum(tuple(channels.values()), size), forcing, "explicit source channel difference")

        def work(values):
            result = _component_budget(integration.balance, values)
            return {**result, "variance_work": DT*result["variance_rate"],
                    "weighted_total_work": DT*result["weighted_total_rate"]}

        balance = integration.balance
        child_mean = balance.mean
        boundary_self = -reference.epi_weight*sum(
            (w*(entry_delta[i]-child_mean)**2 for i, _j, w in balance.cut_edges), F(0))
        boundary_input = reference.epi_weight*sum(
            (w*(entry_delta[i]-child_mean)*(entry_delta[j]-child_mean)
             for i, j, w in balance.cut_edges), F(0))
        _equal(boundary_self+boundary_input, balance.variance_boundary_rate,
               "boundary self-relaxation and incoming field split")

        glyphs = tuple([row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"] for trace in traces)
        same_order = tuple(row["node"] for row in glyphs[0]) == tuple(row["node"] for row in glyphs[1])
        local, gaps, local_pressure, cursor = [], [], [], gen_delta
        if same_order:
            for left, right in zip(*glyphs, strict=True):
                if left["outcome"] != "completed" or right["outcome"] != "completed":
                    raise ValueError("completed step has incomplete glyph receipt")
                for row in (left, right):
                    _domain(row["before"], reference)
                    _domain(row["after"], reference)
                    if any(F(row[key]["state"]["time"]) != expected_time for key in ("before", "after")):
                        raise ValueError("preintegration glyph clock differs")
                before = _sub(_v(right["before"]["state"]["epi"]), _v(left["before"]["state"]["epi"]))
                after = _sub(_v(right["after"]["state"]["epi"]), _v(left["after"]["state"]["epi"]))
                pressure_write = _sub(
                    _sub(_v(right["after"]["state"]["pressure"]), _v(right["before"]["state"]["pressure"])),
                    _sub(_v(left["after"]["state"]["pressure"]), _v(left["before"]["state"]["pressure"])))
                local_pressure.append(pressure_write)
                gaps.append(asdict(budget(cursor, before)))
                local.append({"node": left["node"], "control_glyph": left["glyph"],
                              "perturbed_glyph": right["glyph"], "budget": asdict(budget(before, after)),
                              "pressure_write": work(pressure_write)})
                cursor = after
            gaps.append(asdict(budget(cursor, entry_delta)))
            for field in ("mass_change", "variance_change"):
                _equal(sum((row["budget"][field] for row in local), F(0)) +
                       sum((row[field] for row in gaps), F(0)), getattr(staged["pre_integration"], field),
                       "per-glyph and intervening EPI telescope")
        direct = budget(before_delta, final_delta)
        for field in ("mass_change", "variance_change"):
            _equal(sum((getattr(item, field) for item in staged.values()), F(0)),
                   getattr(direct, field), "complete paired step telescope")
        budgets.append({"ordinal": ordinal, "start_time": expected_time, "end_time": expected_time+DT,
                        "budgets": {key: asdict(value) for key, value in staged.items()},
                        "pressure_parts": {key: work(value) for key, value in pressure_parts.items()},
                        "source_channels": {key: work(value) for key, value in channels.items()},
                        "boundary_split": {"self_relaxation_rate": boundary_self,
                                           "incoming_field_rate": boundary_input,
                                           "self_relaxation_work": DT*boundary_self,
                                           "incoming_field_work": DT*boundary_input,
                                           "scope": "Algebraic parent-boundary transport split at the entry state; the negative self term is not an autonomous maintenance law."},
                        "local_glyphs_available": same_order, "local_glyphs": local, "intervening_epi_budgets": gaps,
                        "unassigned_pressure_write": work(_sub(pressure_parts["pressure_operator_write"],
                                                               _sum(local_pressure, size))) if same_order else None,
                        "mass_change": direct.mass_change, "variance_change": direct.variance_change,
                        "scope": "Exact finite paired accounting; source and pressure work are signed Euler terms, not integrated fluxes. "
                                 "Aligned local receipts do not imply common glyphs or a common autonomous policy."})
        current, previous_records = final, tuple(step["endpoint"] for step in pair)
        endpoints.append({"time": expected_time+DT,
                          **endpoint_readout(*final, reference, children, initial_control_epi)})
    total = sum((step["variance_change"] for step in budgets), F(0))
    expected = endpoints[-1]["paired"]["centered_H_energy"]-endpoints[0]["paired"]["centered_H_energy"]
    _equal(total, expected, "whole finite child error telescope")
    return {"children": children, "nodes": source.nodes, "full_metric_weights": reference.metric_weights,
            "endpoints": endpoints, "steps": budgets, "completed_pairs": len(budgets),
            "variance_change": total, "variance_telescope_residual": total-expected,
            "decision": classify_response(endpoints, complete=len(budgets) == COUNT),
            "autonomous_maintenance_certified": False}
