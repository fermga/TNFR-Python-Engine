"""Independent transmission derivatives and shared retained evidence.

Static fields/Jacobians, one-step latency and protocol admission are tested
without repeating the nine retained trajectories.
"""

import json
import math
from copy import deepcopy
from fractions import Fraction as Q

import numpy as np
import pytest

from benchmarks import relational_region_interaction as campaign
from tnfr.dynamics.relational import (
    evaluate_relational_exchange,
    step_relational_exchange,
)


def _exact(item):
    if set(item) == {"numerator", "denominator"}:
        return Q(item["numerator"], item["denominator"])
    return item


@pytest.fixture(scope="module")
def retained():
    directory = campaign.ROOT / "docs/assets/relational_region_interaction"
    return tuple(
        json.loads((directory / name).read_text(encoding="utf-8"), object_hook=_exact)
        for name in ("result.prediction.json", "result.json")
    )


def _initial_acceleration(impulse_node):
    graph = campaign._graph("joined", impulse_node=impulse_node)
    field = evaluate_relational_exchange(graph, model=campaign.MODEL)
    size = len(graph)
    laplacian = np.zeros((size, size))
    hessian = np.zeros((size, size))
    resultant = np.zeros(size)
    degrees = np.zeros(size)
    for i, j in graph.edges:
        cosine = math.cos(campaign.REFERENCE_PHASE[j] - campaign.REFERENCE_PHASE[i])
        for matrix, weight in ((laplacian, 1.0), (hessian, cosine)):
            matrix[i, i] += weight
            matrix[j, j] += weight
            matrix[i, j] -= weight
            matrix[j, i] -= weight
        resultant[i] += cosine
        resultant[j] += cosine
        degrees[i] += 1
        degrees[j] += 1
    q = laplacian @ np.array(field.epi)
    # The ideal initial phase gradient is zero. The represented pressure keeps
    # its round-off residual, checked independently against this ideal row.
    form_rate = -0.5 * q / degrees
    phase_rate = 0.5 * q / (math.pi * resultant)
    assert field.form_rate == pytest.approx(form_rate, abs=1e-15)
    assert field.phase_rate == pytest.approx(phase_rate, abs=1e-15)
    form_acceleration = -0.5 * (laplacian @ form_rate) / degrees
    form_acceleration -= 0.5 * (hessian @ phase_rate) / (math.pi * resultant)
    # On the receiver q=0, so differentiating H(theta)^(-1) adds no term.
    phase_acceleration = 0.5 * (laplacian @ form_rate) / (math.pi * resultant)
    return graph, field, form_acceleration[5:], phase_acceleration[5:]


def test_internal_impulse_has_predicted_second_order_receiver_response():
    graph, field, form, phase = _initial_acceleration(1)
    epsilon = float(campaign.EPSILON)
    h = 1 + 2 * math.cos(math.tau / 5)
    expected_form = epsilon * (1 / 36 - 1 / (4 * math.pi**2 * h**2))
    expected_phase = -epsilon / (12 * math.pi * h)
    assert form == pytest.approx((expected_form, 0, 0, 0, 0), abs=1e-18)
    assert phase == pytest.approx((expected_phase, 0, 0, 0, 0), abs=1e-18)
    assert expected_form > 0 and expected_phase < 0
    assert max(map(abs, field.form_rate[5:] + field.phase_rate[5:])) < 1e-15
    # A directional derivative of the actual native field checks the analytic
    # receiver Jacobian. These two detached state probes are not solver steps.
    delta = 2**-14
    samples = []
    for sign in (-1, 1):
        probe = deepcopy(graph)
        for i in graph:
            probe.nodes[i]["EPI"] += sign * delta * field.form_rate[i]
            probe.nodes[i]["theta"] += sign * delta * field.phase_rate[i]
        rates = evaluate_relational_exchange(probe, model=campaign.MODEL)
        samples.append(np.array(rates.form_rate[5:] + rates.phase_rate[5:]))
    observed = (samples[1] - samples[0]) / (2 * delta)
    assert observed == pytest.approx(np.concatenate((form, phase)), abs=2e-11)


def test_equal_mean_storage_and_phase_summaries_do_not_close_receiver_dynamics():
    _, near, near_form, near_phase = _initial_acceleration(1)
    _, remote, remote_form, remote_phase = _initial_acceleration(2)
    assert sum(map(Q, near.epi[:5])) == sum(map(Q, remote.epi[:5]))
    assert near.form_storage == remote.form_storage
    assert near.phase_storage == remote.phase_storage
    assert near.storage == remote.storage
    assert near.phase == remote.phase
    assert remote_form == pytest.approx(np.zeros(5), abs=1e-18)
    assert remote_phase == pytest.approx(np.zeros(5), abs=1e-18)
    assert near_form[0] > 0 and near_phase[0] < 0
    # Spatial position and port-neighbor information were discarded by those
    # summaries; this does not exclude reductions retaining boundary data.
    assert near.epi != remote.epi


def test_initial_snapshot_reuses_exact_cut_accounting_and_independent_source():
    graph = campaign._graph("joined")
    field = evaluate_relational_exchange(graph, model=campaign.MODEL)
    before = deepcopy(dict(graph.nodes(data=True)))
    snapshot = campaign._snapshot(graph, field, time=0.0)
    assert dict(graph.nodes(data=True)) == before
    for region in ("left", "right"):
        balance = snapshot["regional_transport"][region]
        assert balance["mass_identity_residual"] == 0
        assert balance["variance_identity_residual"] == 0
        assert balance["outward_cut_current"] == 0
    assert snapshot["regional_transport"]["left"]["mean"] == 2 * campaign.EPSILON / 11
    assert snapshot["regional_transport"]["right"]["mean"] == 0
    assert max(map(abs, field.phase_source)) < 1e-15


def test_euler_first_step_does_not_invent_an_instant_receiver_response():
    graph = campaign._graph("joined")
    step = step_relational_exchange(graph, model=campaign.MODEL, dt=1 / 4096)
    assert max(map(abs, step.after.epi[5:])) < 1e-18
    assert step.after.phase[5:] == step.before.phase[5:]
    # Subsequent rates respond after the simultaneous left update. This is an
    # explicit Euler property, not a claim of a physical propagation delay.
    assert step.after.form_rate[5] > 0
    assert step.after.phase_rate[5] < 0


def test_changed_protocol_rejects_before_any_trace(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unfrozen trajectory execution")

    monkeypatch.setattr(campaign, "_trace", forbidden)
    prediction = campaign.prepare_prediction()
    for key, value in (
        ("capacity", (True,) + (1.0,) * 9),
        ("step_counts", (64, 128, 512)),
    ):
        changed = deepcopy(prediction)
        changed[key] = value
        with pytest.raises(ValueError, match="frozen protocol"):
            campaign.evaluate_prediction(changed)
    prediction["regions"]["right"] = ()
    assert campaign.prepare_prediction()["regions"]["right"] == tuple(range(5, 10))


def test_cli_requires_frozen_protocol_and_exclusive_response(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("trajectory was not admitted")

    monkeypatch.setattr(campaign, "evaluate_prediction", forbidden)
    with pytest.raises(FileNotFoundError):
        campaign.main(["--output", str(output)])
    assert campaign.main(["--prepare", "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        campaign.main(["--prepare", "--output", str(output)])
    output.write_text("retained", encoding="utf-8")
    with pytest.raises(FileExistsError, match="retained"):
        campaign.main(["--output", str(output)])
    output.unlink()
    output.with_suffix(".prediction.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze"):
        campaign.main(["--output", str(output)])


def test_retained_response_and_refinement_are_recounted(retained):
    prediction, report = retained
    assert report["prediction"] == prediction
    assert set(report["traces"]) == {"joined", "left", "right"}
    for name in ("form", "phase"):
        differences = []
        for count in campaign.STEP_COUNTS:
            joined = report["traces"]["joined"][str(count)]["checkpoints"][str(count)][
                "regions"
            ]["right"]
            control = report["traces"]["right"][str(count)]["checkpoints"][str(count)][
                "regions"
            ]["right"]
            difference = joined[f"chi_{name}"] - control[f"chi_{name}"]
            differences.append(difference)
            row = report["responses"][str(count)][name]
            assert row["difference"] == difference
            ratio = float(difference) / prediction["leading_response_estimates"][name]
            assert row["normalized_response"] == ratio
            assert 0.75 <= ratio <= 1.25
        first, last = abs(differences[1] - differences[0]), abs(
            differences[2] - differences[1]
        )
        refinement = report["refinements"][name]
        assert refinement["coarse_to_middle"] == first
        assert refinement["middle_to_fine"] == last
        assert refinement["contracting"] == (last <= 0.75 * first + 1e-12)
        assert refinement["signal_resolved"] == (
            abs(differences[-1]) > 32 * (last + 1e-12)
        )
    assert report["passed"] and all(report["checks"].values())


def test_retained_checkpoints_replay_fields_geometry_and_cut_accounting(retained):
    _, report = retained
    for scenario, traces in report["traces"].items():
        for count in campaign.STEP_COUNTS:
            trace = traces[str(count)]
            assert trace["observed_state_count"] == count + 1
            assert set(trace["checkpoints"]) == {"0", str(count // 2), str(count)}
            graph = campaign._graph(scenario)
            initial = trace["checkpoints"]["0"]
            for key, attribute in (
                ("epi", "EPI"),
                ("phase", "theta"),
                ("capacity", "nu_f"),
            ):
                assert initial[key] == [graph.nodes[node][attribute] for node in graph]
            for index, saved in trace["checkpoints"].items():
                assert saved["time"] == int(index) * trace["dt"]
                assert saved["nodes"] == list(graph)
                assert saved["capacity"] == [1.0] * len(graph)
                for i, node in enumerate(graph):
                    graph.nodes[node].update(
                        EPI=saved["epi"][i],
                        theta=saved["phase"][i],
                        nu_f=saved["capacity"][i],
                    )
                field = evaluate_relational_exchange(graph, model=campaign.MODEL)
                for name in ("pressure", "form_rate", "phase_rate", "phase_source"):
                    assert getattr(field, name) == pytest.approx(saved[name], abs=1e-12)
                replay = campaign._snapshot(graph, field, time=saved["time"])
                for name, region in saved["regions"].items():
                    rebuilt = json.loads(
                        campaign._encoded(replay["regions"][name]), object_hook=_exact
                    )
                    assert rebuilt == region
                    nodes = campaign.REGIONS[name]
                    winding = (
                        math.fsum(
                            math.remainder(
                                graph.nodes[nodes[(i + 1) % 5]]["theta"]
                                - graph.nodes[nodes[i]]["theta"],
                                math.tau,
                            )
                            for i in range(5)
                        )
                        / math.tau
                    )
                    assert winding == pytest.approx(1.0, abs=1e-15)
                for balance in saved.get("regional_transport", {}).values():
                    assert balance["mass_identity_residual"] == 0
                    assert balance["variance_identity_residual"] == 0
                for name, balance in replay.get("regional_transport", {}).items():
                    for key, value in balance.items():
                        assert float(value) == pytest.approx(
                            float(saved["regional_transport"][name][key]), abs=1e-12
                        )
