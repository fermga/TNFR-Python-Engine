"""Static response controls for the complete sine law, without trajectories."""

import json
import pickle
from dataclasses import replace
from fractions import Fraction as Q
from math import isqrt

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import certify_sine_pattern_recovery
from tnfr.physics.relational_sine_resonance import (
    assess_sine_cycle_resonance,
    assess_sine_mediated_response,
    assess_sine_pair_pulse,
    assess_sine_path_memory,
    assess_sine_recurrence,
    certify_sine_recovery_resonance,
)
from tnfr.sdk import export_to_json, relational_report_to_dict


def _model(beta=1, **kwargs):
    return RelationalExchangeModel(beta, phase_domain="regular", **kwargs)


def _mode(**kwargs):
    arguments = dict(model=_model(), node_count=5, mode_index=1, capacity=1)
    arguments.update(kwargs)
    return assess_sine_cycle_resonance(**arguments)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert bound is not None
    if bound.lo == bound.hi:
        assert mp.almosteq(value, _mp(bound.lo), abs_eps=mp.mpf("1e-80"))
    else:
        assert _mp(bound.lo) <= value <= _mp(bound.hi)


def test_cycle_retains_exact_stored_coefficients_without_normalization():
    model = _model()
    for field, value in (
        ("epi_weight", Q(7, 19)),
        ("phase_weight", Q(5, 17)),
        ("storage_scale", Q(11, 13)),
    ):
        object.__setattr__(model, field, value)
    report = _mode(model=model, node_count=4, mode_index=2, capacity=Q(3, 2))
    assert report.model is model
    assert report.work_peak_gain == Q(76, 21)
    with mp.workdps(90):
        # The alternating C4 mode has exact Laplacian eigenvalue 4.
        generator = report.modal_generator_bounds
        _contains(generator[0][0], -mp.mpf(21) / 19)
        _contains(generator[0][1], -mp.mpf(15) / (17 * mp.pi))
        _contains(generator[1][0], mp.mpf(195) / (187 * mp.pi))
        _contains(generator[1][1], mp.mpf(0))


def test_mode_gain_rebuilds_cached_generator_and_transfer_bounds():
    from tnfr.mathematics._rational_interval import I

    source = _mode(node_count=4, mode_index=2)
    changed = replace(
        source,
        modal_generator_bounds=((I(0), I(-1)), (I(0), I(0))),
        damping_coefficient_bounds=I(0),
        stiffness_bounds=I(1),
        laplacian_eigenvalue_bounds=I(0),
        pole_status="unresolved",
    )
    gain = changed.gain(Q(1, 10))
    assert gain == source.gain(Q(1, 10))
    assert gain.mode == source and gain.mode is not changed
    assert gain.phase_gain_bounds.lo > 0
    assert gain.work_gain_bounds.lo > 0


def test_mode_gain_rebuilds_changed_capacity_under_the_declared_clock():
    source = _mode(node_count=4, mode_index=2)
    gain = replace(source, capacity=Q(2)).gain(0)
    assert gain.mode == _mode(node_count=4, mode_index=2, capacity=2)
    assert gain.mode.work_peak_gain == 2
    with mp.workdps(90):
        # C4 alternating mode has lambda=4; b=1/(2*pi), r=nu*lambda/2=4.
        _contains(gain.phase_gain_bounds, mp.pi / 2)


@pytest.mark.parametrize(
    "changes",
    (
        {"node_count": True},
        {"node_count": 2},
        {"mode_index": True},
        {"mode_index": 0},
        {"winding": True},
        {"winding": 2},
        {"capacity": True},
        {"capacity": -1},
        {"capacity": float("nan")},
        {"model": RelationalExchangeModel(1)},
        {"model": RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")},
        {"law": "current_squared_reciprocal_mobility"},
    ),
)
def test_mode_gain_readmits_authoritative_mode_and_law(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(_mode(), **changes).gain(Q(1, 10))


def test_mode_gain_readmits_stored_model_coefficients():
    source = _mode()
    model = _model()
    object.__setattr__(model, "storage_scale", True)
    with pytest.raises((TypeError, ValueError)):
        replace(source, model=model).gain(Q(1, 10))


def _laplacian(graph, weights=None):
    matrix = mp.matrix(len(graph))
    for i, j in graph.edges:
        weight = mp.mpf(1) if weights is None else weights(i, j)
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight
        matrix[j, i] -= weight
    return matrix


def _cycle_matrix(report):
    n, mode, winding = report.node_count, report.mode_index, report.winding
    graph = nx.cycle_graph(n)
    vector = mp.matrix([mp.cos(2 * mp.pi * mode * j / n) for j in graph])
    vector /= mp.norm(vector)
    phase = mp.matrix([2 * mp.pi * winding * j / n for j in graph])
    laplacian = _laplacian(graph)
    hessian = _laplacian(graph, lambda i, j: mp.cos(phase[j] - phase[i]))
    e, w = map(_mp, report.model.effective_weights)
    a, b = w / mp.pi, w / (_mp(report.model.storage_scale) * mp.pi)
    mobility = _mp(report.capacity) / 2
    full = mp.matrix(2 * n)
    for i in graph:
        for j in graph:
            full[i, j] = -e * mobility * laplacian[i, j]
            full[i, n + j] = -a * mobility * hessian[i, j]
            full[n + i, j] = b * mobility * laplacian[i, j]
    return graph, vector, phase, laplacian, full


@pytest.mark.parametrize("n,mode,winding", [(4, 2, 0), (5, 1, 1), (7, 2, -1)])
def test_modal_generator_differentiates_full_nonlinear_cycle_field(n, mode, winding):
    report = _mode(node_count=n, mode_index=mode, winding=winding, capacity=Q(3, 2))
    with mp.workdps(100):
        graph, vector, phase, laplacian, full = _cycle_matrix(report)
        e, w = map(_mp, report.model.effective_weights)
        mobility = _mp(report.capacity) / 2

        def projected_field(x_amplitude, phase_amplitude):
            x = vector * x_amplitude
            theta = phase + vector * phase_amplitude
            q = laplacian * x
            current = mp.matrix(
                [sum(mp.sin(theta[j] - theta[i]) for j in graph[i]) for i in graph]
            )
            form = mobility * (-e * q + w / mp.pi * current)
            phase_rate = mobility * w / (_mp(report.model.storage_scale) * mp.pi) * q
            return (vector.T * form)[0], (vector.T * phase_rate)[0]

        for column in range(2):
            direction = (
                mp.matrix(list(vector) + [0] * n)
                if column == 0
                else mp.matrix([0] * n + list(vector))
            )
            image = full * direction
            for row in range(2):
                derivative = mp.diff(
                    lambda t: (
                        projected_field(t, 0)[row]
                        if column == 0
                        else projected_field(0, t)[row]
                    ),
                    0,
                )
                projected = sum(vector[i] * image[row * n + i] for i in graph)
                assert abs(derivative - projected) < mp.mpf("1e-90")
                _contains(report.modal_generator_bounds[row][column], derivative)
        _contains(
            report.laplacian_eigenvalue_bounds, (vector.T * laplacian * vector)[0]
        )


@pytest.mark.parametrize(
    "beta,poles,phase_peak",
    [
        (1, "distinct_real_poles", "dc_maximum"),
        (Q(1, 4), "complex_conjugate_poles", "dc_maximum"),
        (Q(1, 8), "complex_conjugate_poles", "positive_frequency_peak"),
    ],
)
def test_full_resolvent_separates_free_poles_and_each_observed_gain(
    beta, poles, phase_peak
):
    report = _mode(model=_model(beta))
    assert report.pole_status == poles
    assert report.phase_peak_status == phase_peak
    assert report.work_peak_gain == 4
    with mp.workdps(100):
        graph, vector, _, laplacian, full = _cycle_matrix(report)
        n = len(graph)
        force = mp.matrix(list(vector) + [0] * n)
        eigenvalue = (vector.T * laplacian * vector)[0]
        e, w = map(_mp, report.model.effective_weights)
        r = eigenvalue / 2
        feedback = w**2 / (_mp(beta) * mp.pi**2)
        natural = r * mp.sqrt(feedback)
        _contains(report.natural_angular_frequency_bounds, natural)
        _contains(report.form_peak_gain_bounds, 1 / (e * r))
        peak_response = mp.lu_solve(mp.j * natural * mp.eye(2 * n) - full, force)
        peak_form = abs(sum(vector[i] * peak_response[i] for i in graph))
        assert mp.almosteq(peak_form * eigenvalue, 4)
        for frequency in (Q(1, 10), Q(1, 3), Q(2)):
            gain = report.gain(frequency)
            response = mp.lu_solve(mp.j * _mp(frequency) * mp.eye(2 * n) - full, force)
            form = abs(sum(vector[i] * response[i] for i in graph))
            phase = abs(sum(vector[i] * response[n + i] for i in graph))
            _contains(gain.form_gain_bounds, form)
            _contains(gain.phase_gain_bounds, phase)
            _contains(gain.work_gain_bounds, eigenvalue * form)
            assert form < peak_form
        dc = report.gain(0)
        assert dc.form_gain_bounds.lo == dc.form_gain_bounds.hi == 0
        assert dc.work_gain_bounds.lo == dc.work_gain_bounds.hi == 0
        _contains(dc.phase_gain_bounds, 1 / ((w / mp.pi) * r))
        if phase_peak == "positive_frequency_peak":
            frequency = r * mp.sqrt(feedback - e**2 / 2)
            _contains(report.phase_peak_angular_frequency_bounds, frequency)
            response = mp.lu_solve(mp.j * frequency * mp.eye(2 * n) - full, force)
            actual = abs(sum(vector[i] * response[n + i] for i in graph))
            _contains(report.phase_peak_gain_bounds, actual)
            assert actual > _mp(dc.phase_gain_bounds.hi)
        else:
            assert report.phase_peak_gain_bounds == report.phase_dc_gain_bounds


def test_capacity_and_joint_form_unit_clock_covariance_preserve_classification():
    old = _mode(model=_model(Q(1, 8)), winding=1)
    faster = _mode(model=old.model, winding=1, capacity=3)
    transformed = _mode(
        model=_model(Q(9, 8), epi_weight=1, phase_weight=3), winding=1, capacity=Q(1, 2)
    )
    reverse = _mode(model=old.model, winding=-1, mode_index=4)
    assert old.modal_generator_bounds == reverse.modal_generator_bounds
    assert old.constitutive_ratio == transformed.constitutive_ratio
    assert old.pole_status == faster.pole_status == transformed.pole_status
    assert (
        old.phase_peak_status
        == faster.phase_peak_status
        == transformed.phase_peak_status
    )
    with mp.workdps(100):
        graph, vector, _, laplacian, full = _cycle_matrix(old)
        n = len(graph)
        for i in range(2):
            for j in range(2):
                projected = sum(
                    vector[k] * full[i * n + k, j * n + ell] * vector[ell]
                    for k in graph
                    for ell in graph
                )
                _contains(faster.modal_generator_bounds[i][j], 3 * projected)
                _contains(
                    transformed.modal_generator_bounds[i][j],
                    projected * ([3, 1][i] / mp.mpf([3, 1][j])) / 4,
                )
        # y=3*x and tau=4*t, with normalization absorbed into held capacity.
        scaled = transformed.gain(Q(1, 12))
        response = mp.lu_solve(
            mp.j / 3 * mp.eye(2 * n) - full,
            mp.matrix(list(vector) + [0] * n),
        )
        form = abs(sum(vector[i] * response[i] for i in graph))
        phase = abs(sum(vector[i] * response[n + i] for i in graph))
        eigenvalue = (vector.T * laplacian * vector)[0]
        for name, actual, factor in (
            ("form_gain_bounds", form, 4),
            ("phase_gain_bounds", phase, Q(4, 3)),
            ("work_gain_bounds", eigenvalue * form, 4),
        ):
            _contains(getattr(scaled, name), _mp(factor) * actual)
        assert faster.work_peak_gain == old.work_peak_gain / 3


@pytest.mark.parametrize(
    "beta,field", [(Q(1, 4), "pole_status"), (Q(1, 8), "phase_peak_status")]
)
def test_unresolved_threshold_is_not_declared_exact_critical_damping_or_dc_peak(
    beta, field
):
    with mp.workdps(120):
        count = 10**80
        winding = int(mp.nint(count * mp.acos(mp.pi**2 / 16) / (2 * mp.pi)))
    report = _mode(
        model=_model(beta), node_count=count, mode_index=count // 4, winding=winding
    )
    assert getattr(report, field) == "unresolved"
    if field == "pole_status":
        assert report.damped_angular_frequency_bounds is None
    else:
        assert report.phase_peak_angular_frequency_bounds is None
        assert report.phase_peak_gain_bounds is None


def test_tiny_exact_capacity_and_frequency_never_become_false_zero_response():
    tiny = Q(1, 2**200)
    report = _mode(capacity=tiny)
    assert report.capacity == tiny
    assert report.pole_status == "distinct_real_poles"
    assert report.work_peak_gain == 4 / tiny
    zero, nonzero = report.gain(0), report.gain(tiny)
    assert zero.form_gain_bounds.lo == zero.form_gain_bounds.hi == 0
    assert nonzero.angular_frequency == tiny
    assert nonzero.status == "unavailable"
    assert (
        nonzero.form_gain_bounds
        is nonzero.phase_gain_bounds
        is nonzero.work_gain_bounds
        is None
    )
    assert "positive_transfer_denominator_not_resolved" in nonzero.unavailable_reasons


@pytest.mark.parametrize(
    "changes",
    [
        {"node_count": 2},
        {"node_count": True},
        {"mode_index": 0},
        {"mode_index": 5},
        {"mode_index": 1.0},
        {"winding": Q(1, 2)},
        {"node_count": 4, "winding": 1},
        {"capacity": 0},
        {"capacity": -1},
        {"capacity": True},
        {"capacity": float("nan")},
        {"model": RelationalExchangeModel(1)},
        {"model": _model(epi_weight=0)},
    ],
)
def test_cycle_rejects_unsupported_domains_and_scalar_representations(changes):
    with pytest.raises((TypeError, ValueError)):
        _mode(**changes)


def test_frequency_admission_and_exact_export_use_shared_report_boundary(tmp_path):
    report = _mode(capacity=Q(7, 8), winding=1)
    for bad in (-1, True, float("inf"), "0.1", complex(1, 0)):
        with pytest.raises((TypeError, ValueError)):
            report.gain(bad)
    gain = report.gain(Q(1, 10))
    for owner, schema in (
        (report, "tnfr.relational-sine-cycle-resonance.v1"),
        (gain, "tnfr.relational-sine-mode-gain.v1"),
    ):
        payload = owner.to_dict()
        assert payload["schema"] == schema
        assert relational_report_to_dict(owner)["report"] == payload["report"]
        path = tmp_path / (schema + ".json")
        export_to_json(payload, path)
        assert json.loads(path.read_text()) == payload
    assert report.to_dict()["report"]["capacity"] == {"numerator": 7, "denominator": 8}


def _recovery(*, error=0):
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 5)
    for i in graph:
        graph.nodes[i].update(EPI=0, theta=1287 * i / 1024 if i < 5 else 0, nu_f=1)
    before = pickle.dumps(
        (graph.graph, dict(graph.nodes(data=True)), dict(graph.edges))
    )
    pattern = bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=_model(),
        form_error_bounds=(error,) * 6,
        phase_error_bounds=(error,) * 6,
    )
    recovery = certify_sine_pattern_recovery(
        pattern,
        target_phase_turns=tuple(Q(j, 5) for j in range(5)) + (Q(0),),
        radius=Q(1, 16),
    )
    assert (
        pickle.dumps((graph.graph, dict(graph.nodes(data=True)), dict(graph.edges)))
        == before
    )
    return graph, recovery


def test_full_noncommuting_support_has_collocated_gain_bound_without_cycle_reduction():
    graph, recovery = _recovery()
    assert recovery.admitted
    report = certify_sine_recovery_resonance(recovery, port=(1, 5))
    assert report.positive_frequency_peak_certified
    assert report.peak_angular_frequency_bounds is None
    assert not report.capacity_family and not report.input_output_family
    assert report.work_gain_upper_bound == 3
    with mp.workdps(100):
        n = len(graph)
        phase = [2 * mp.pi * j / 5 for j in range(5)] + [mp.mpf(0)]
        laplacian = _laplacian(graph)
        hessian = _laplacian(graph, lambda i, j: mp.cos(phase[j] - phase[i]))
        mobility = mp.diag([mp.mpf(1) / graph.degree[i] for i in graph])
        root = mp.diag([mp.sqrt(mobility[i, i]) for i in graph])
        bmatrix, cmatrix = root * laplacian * root, root * hessian * root
        assert mp.norm(bmatrix * cmatrix - cmatrix * bmatrix) > mp.mpf("0.1")
        force = mp.matrix([0] * n)
        force[1], force[5] = mobility[1, 1], -mobility[5, 5]
        output = laplacian * force
        for i in graph:
            _contains(report.input_coefficients_bounds[i], force[i])
            _contains(report.work_output_coefficients_bounds[i], output[i])
        assert abs(sum(force[i] / mobility[i, i] for i in graph)) == 0
        e, a, b = mp.mpf("0.5"), 1 / (2 * mp.pi), 1 / (2 * mp.pi)
        full = mp.matrix(2 * n)
        for i in graph:
            for j in graph:
                full[i, j] = -e * mobility[i, i] * laplacian[i, j]
                full[i, n + j] = -a * mobility[i, i] * hessian[i, j]
                full[n + i, j] = b * mobility[i, i] * laplacian[i, j]
        for frequency in (Q(1, 10), Q(1, 3), Q(1), Q(3)):
            response = mp.lu_solve(
                mp.j * _mp(frequency) * mp.eye(2 * n) - full,
                mp.matrix(list(force) + [0] * n),
            )
            x = mp.matrix(list(response)[:n])
            gain = (output.T * x)[0]
            q = laplacian * x
            dissipation = e * (q.conjugate().T * mobility * q)[0]
            assert mp.almosteq(gain.real, dissipation.real)
            assert gain.real > 0 and 0 < abs(gain) <= _mp(report.work_gain_upper_bound)
        # A common-origin input is invisible to this collocated quotient output.
        assert mp.norm(laplacian * mp.matrix([1] * n)) == 0
    assert report.to_dict()["schema"] == "tnfr.relational-sine-recovery-resonance.v1"
    assert relational_report_to_dict(report)["report"] == report.to_dict()["report"]


def test_recovery_port_revalidates_admission_and_rejects_forged_certificates():
    _, recovery = _recovery()
    _, unavailable = _recovery(error=1)
    assert not unavailable.admitted
    for bad in (unavailable, replace(recovery, radius=Q(1, 32))):
        with pytest.raises(ValueError):
            certify_sine_recovery_resonance(bad, port=(1, 5))
    with pytest.raises(TypeError):
        certify_sine_recovery_resonance(object(), port=(1, 5))
    for port in ((0,), (0, 1, 2), (1, 1), (1, 99), "01", {1, 5}):
        with pytest.raises((TypeError, ValueError)):
            certify_sine_recovery_resonance(recovery, port=port)


def _mediated_recovery(*, capacities=None, extra_edge=False, winding_signs=(1, 1)):
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    for start in (0, 5):
        graph.add_edges_from((start + j, start + (j + 1) % 5) for j in range(5))
    graph.add_edges_from(((0, 10), (5, 10)))
    if extra_edge:
        graph.add_edge(0, 5)
    capacities = (Q(1),) * 11 if capacities is None else capacities
    target = tuple(sign * Q(j, 5) for sign in winding_signs for j in range(5)) + (Q(0),)
    for i in graph:
        graph.nodes[i].update(
            EPI=0,
            theta=winding_signs[i // 5] * 1287 * (i % 5) / 1024 if i != 10 else 0,
            nu_f=capacities[i],
        )
    pattern = bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=_model(),
        form_error_bounds=(0,) * 11,
        phase_error_bounds=(0,) * 11,
    )
    recovery = certify_sine_pattern_recovery(
        pattern,
        target_phase_turns=target,
        radius=Q(1, 16),
    )
    return graph, recovery


def _remote_report(recovery, **changes):
    arguments = dict(
        donor_cycle=tuple(range(5)), receiver_cycle=tuple(range(5, 10)), mediator=10
    )
    arguments.update(changes)
    return assess_sine_mediated_response(recovery, **arguments)


def _remote_matrices(graph, capacities, model):
    n = len(graph)
    phase = mp.matrix([2 * mp.pi * (i % 5) / 5 if i != 10 else 0 for i in graph])
    laplacian = _laplacian(graph)
    hessian = _laplacian(graph, lambda i, j: mp.cos(phase[j] - phase[i]))
    mobility = mp.diag([_mp(capacities[i]) / graph.degree[i] for i in graph])
    e, w = map(_mp, model.effective_weights)
    a, b = w / mp.pi, w / (_mp(model.storage_scale) * mp.pi)
    generator = mp.matrix(2 * n)
    for i in graph:
        for j in graph:
            generator[i, j] = -e * mobility[i, i] * laplacian[i, j]
            generator[i, n + j] = -a * mobility[i, i] * hessian[i, j]
            generator[n + i, j] = b * mobility[i, i] * laplacian[i, j]
    donor = mp.matrix([1] + [mp.mpf(-1) / 4] * 4 + [0] * 6)
    receiver = mp.matrix([0] * 5 + [1] + [mp.mpf(-1) / 4] * 4 + [0])
    direction = mobility * donor
    return phase, laplacian, hessian, mobility, generator, donor, receiver, direction


@pytest.mark.parametrize(
    "capacities",
    [
        (Q(1),) * 11,
        tuple(map(Q, (1, 2, 3, 4, 5, 1, 3, 2, 4, 2))) + (Q(3, 2),),
    ],
)
def test_remote_markov_rows_match_complete_generator_with_held_capacities(capacities):
    graph, recovery = _mediated_recovery(capacities=capacities)
    report = _remote_report(recovery)
    assert report.positive_frequency_phase_peak_certified
    assert report.phase_impulse_sign_reversal_certified
    assert (
        report.peak_angular_frequency_bounds is report.sign_reversal_time_bounds is None
    )
    if capacities == (Q(1),) * 11:
        _, opposite = _mediated_recovery(winding_signs=(-1, 1))
        opposite_response = _remote_report(opposite)
        assert opposite.target_phase_turns != recovery.target_phase_turns
        assert (
            opposite_response.form_state_derivatives_bounds
            == report.form_state_derivatives_bounds
        )
        assert (
            opposite_response.phase_state_derivatives_bounds
            == report.phase_state_derivatives_bounds
        )
        assert (
            opposite_response.memory_kernel_at_zero_bounds
            == report.memory_kernel_at_zero_bounds
        )
    with mp.workdps(100):
        _, _, _, mobility, generator, donor, receiver, direction = _remote_matrices(
            graph, capacities, recovery.reference_model
        )
        for i in graph:
            assert _mp(report.donor_contrast[i]) == donor[i]
            assert _mp(report.receiver_contrast[i]) == receiver[i]
            assert mp.almosteq(_mp(report.input_direction[i]), direction[i])
        assert abs(sum(direction[i] / mobility[i, i] for i in graph)) < mp.mpf("1e-90")
        vector = mp.matrix(list(direction) + [0] * 11)
        for order in range(3):
            for i in graph:
                _contains(report.form_state_derivatives_bounds[order][i], vector[i])
                _contains(
                    report.phase_state_derivatives_bounds[order][i], vector[11 + i]
                )
            form = sum(receiver[i] * vector[i] for i in graph)
            phase = sum(receiver[i] * vector[11 + i] for i in graph)
            _contains(report.form_markov_bounds[order], form)
            _contains(report.phase_markov_bounds[order], phase)
            vector = generator * vector
        expected = -mobility[0, 0] * mobility[5, 5] * mobility[10, 10] / (4 * mp.pi)
        _contains(report.phase_markov_bounds[2], expected)
        assert mp.almosteq(_mp(report.phase_order_two_pi_numerator) / mp.pi, expected)
        assert report.phase_order_two_sign == -1


def test_remote_zero_dc_and_memory_schur_complement_preserve_full_response():
    graph, recovery = _mediated_recovery()
    report = _remote_report(recovery)
    with mp.workdps(100):
        _, _, hessian, mobility, generator, donor, receiver, direction = (
            _remote_matrices(graph, (1,) * 11, recovery.reference_model)
        )
        # Fix an arithmetic phase origin only for solving this independent static system.
        steady_phase = mp.lu_solve(hessian + mp.ones(11) / 11, 2 * mp.pi * donor)
        assert mp.norm(hessian * steady_phase - 2 * mp.pi * donor) < mp.mpf("1e-90")
        assert max(
            abs(steady_phase[i] - steady_phase[10]) for i in range(5, 10)
        ) < mp.mpf("1e-90")
        assert abs((receiver.T * steady_phase)[0]) < mp.mpf("1e-90")
        assert report.form_dc_gain == report.phase_dc_gain == 0
        visible = tuple(range(10)) + tuple(range(11, 21))
        hidden = (10, 21)

        def block(rows, columns):
            return mp.matrix([[generator[i, j] for j in columns] for i in rows])

        avv, avh = block(visible, visible), block(visible, hidden)
        ahv, ahh = block(hidden, visible), block(hidden, hidden)
        cross_receiver, cross_donor = block((5, 16), hidden), block(hidden, (0, 11))
        m = mp.matrix([[mp.mpf("0.5"), 1 / (2 * mp.pi)], [-1 / (2 * mp.pi), 0]])
        expected = mobility[5, 5] * mobility[10, 10] * m**2
        assert mp.norm(cross_receiver * cross_donor - expected) < mp.mpf("1e-90")
        for bounds, actual in (
            (report.mediator_generator_bounds, ahh),
            (report.donor_to_mediator_bounds, cross_donor),
            (report.mediator_to_receiver_bounds, cross_receiver),
            (report.memory_kernel_at_zero_bounds, expected),
        ):
            for i in range(2):
                for j in range(2):
                    _contains(bounds[i][j], actual[i, j])
        assert generator[5, 0] == generator[16, 0] == 0
        s = mp.j / 3
        force = mp.matrix(list(direction) + [0] * 11)
        full = mp.lu_solve(s * mp.eye(22) - generator, force)
        hidden_resolvent = (s * mp.eye(2) - ahh) ** -1
        reduced = mp.lu_solve(
            s * mp.eye(20) - avv - avh * hidden_resolvent * ahv,
            mp.matrix([force[i] for i in visible]),
        )
        assert mp.norm(reduced - mp.matrix([full[i] for i in visible])) < mp.mpf(
            "1e-90"
        )
        remote_gain = sum(receiver[i] * full[11 + i] for i in graph)
        assert abs(remote_gain) > mp.mpf("1e-6")
        # This one nonzero sample checks the wiring, not the gain-maximum theorem.


def test_endogenous_nonlinear_initial_acceleration_matches_tangent_derivative():
    graph, recovery = _mediated_recovery()
    report = _remote_report(recovery)
    with mp.workdps(100):
        target, laplacian, _, mobility, generator, _, receiver, direction = (
            _remote_matrices(graph, (1,) * 11, recovery.reference_model)
        )
        epsilon = mp.mpf(1) / 64
        initial = mp.matrix(list(epsilon * direction) + list(target))

        def field(state):
            x, theta = mp.matrix(list(state)[:11]), mp.matrix(list(state)[11:])
            q = laplacian * x
            current = mp.matrix(
                [sum(mp.sin(theta[j] - theta[i]) for j in graph[i]) for i in graph]
            )
            return mp.matrix(
                list(mobility * (-q / 2 + current / (2 * mp.pi)))
                + list(mobility * q / (2 * mp.pi))
            )

        initial_rate = field(initial)
        first = sum(receiver[i] * initial_rate[11 + i] for i in graph)
        second = mp.diff(
            lambda t: sum(
                receiver[i] * field(initial + t * initial_rate)[11 + i] for i in graph
            ),
            0,
        )
        assert abs(first) < mp.mpf("1e-90")
        _contains(report.phase_markov_bounds[2], second / epsilon)
        assert second < 0
        assert (
            abs(
                sum(receiver[i] * initial[11 + i] for i in graph)
                - sum(receiver[i] * target[i] for i in graph)
            )
            == 0
        )
        # A step input from the target integrates the impulse kernel once:
        # its first nonzero phase derivative is order three, not order two.
        force = mp.matrix(list(direction) + [0] * 11)
        step_third = generator**2 * force
        _contains(
            report.phase_markov_bounds[2],
            sum(receiver[i] * step_third[11 + i] for i in graph),
        )


def test_frozen_intermediary_and_reflection_odd_controls_have_no_remote_route():
    graph, recovery = _mediated_recovery()
    report = _remote_report(recovery)
    assert report.donor_odd_tangent_sector_silent
    assert report.receiver_odd_tangent_readout_silent
    with mp.workdps(100):
        _, _, _, mobility, generator, _, receiver, direction = _remote_matrices(
            graph, (1,) * 11, recovery.reference_model
        )
        reflected = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9, 10)
        permutation = mp.matrix(22)
        for offset in (0, 11):
            for i, j in enumerate(reflected):
                permutation[offset + i, offset + j] = 1
        assert mp.norm(permutation * generator - generator * permutation) < mp.mpf(
            "1e-90"
        )
        odd = mp.matrix([0] * 22)
        odd[1], odd[4] = mobility[1, 1], -mobility[4, 4]
        assert mp.norm(permutation * odd + odd) == 0
        output = mp.matrix([0] * 11 + list(receiver)).T
        assert mp.norm(output * permutation - output) == 0
        s = mp.j / 3
        assert abs((output * mp.lu_solve(s * mp.eye(22) - generator, odd))[0]) < mp.mpf(
            "1e-90"
        )
        frozen = generator.copy()
        for row in (10, 21):
            for column in range(22):
                frozen[row, column] = 0
        response = mp.lu_solve(
            s * mp.eye(22) - frozen, mp.matrix(list(direction) + [0] * 11)
        )
        assert max(abs(response[i]) for i in (*range(5, 10), *range(16, 21))) < mp.mpf(
            "1e-90"
        )
        unequal = (1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        other_graph, other = _mediated_recovery(capacities=unequal)
        assert not _remote_report(other).donor_odd_tangent_sector_silent
        _, _, _, k, changed, _, _, _ = _remote_matrices(
            other_graph, unequal, other.reference_model
        )
        odd[1], odd[4] = k[1, 1], -k[4, 4]
        onset = (output * changed**3 * odd)[0]
        expected = (1 / (2 * mp.pi)) * (1 / (4 * mp.pi**2) - mp.mpf(1) / 4)
        expected *= k[0, 0] * k[5, 5] * k[10, 10] * (k[1, 1] - k[4, 4])
        assert mp.almosteq(onset, expected) and abs(onset) > mp.mpf("1e-6")


def test_mediated_reader_rejects_wrong_full_support_and_unrevalidated_sources():
    _, recovery = _mediated_recovery()
    _, extra = _mediated_recovery(extra_edge=True)
    assert (
        extra.admitted
    )  # A zero-turn direct bridge is valid but changes the studied route.
    with pytest.raises(ValueError, match="full support"):
        _remote_report(extra)
    for changes in (
        {"donor_cycle": (0, 1, 2, 3)},
        {"receiver_cycle": (5, 6, 7, 8, 0)},
        {"donor_cycle": (1, 2, 3, 4, 0)},
        {"mediator": 0},
        {"receiver_cycle": (5, 6, 7, 8, 99)},
    ):
        with pytest.raises((TypeError, ValueError)):
            _remote_report(recovery, **changes)
    with pytest.raises(ValueError, match="rebuilt"):
        _remote_report(replace(recovery, radius=Q(1, 32)))
    _, frozen = _mediated_recovery(capacities=(1,) * 10 + (0,))
    assert not frozen.admitted
    with pytest.raises(ValueError, match="admitted"):
        _remote_report(frozen)


def test_remote_exact_tiny_capacity_orientation_and_detached_export(tmp_path):
    tiny = Q(1, 2**200)
    graph, recovery = _mediated_recovery(capacities=(1,) * 10 + (tiny,))
    before = pickle.dumps(
        (graph.graph, dict(graph.nodes(data=True)), dict(graph.edges))
    )
    report = _remote_report(recovery)
    reverse = _remote_report(
        recovery, donor_cycle=(0, 4, 3, 2, 1), receiver_cycle=(5, 9, 8, 7, 6)
    )
    assert report.input_direction == reverse.input_direction
    assert report.receiver_contrast == reverse.receiver_contrast
    assert report.phase_order_two_pi_numerator == -tiny / 72
    assert report.phase_order_two_sign == -1
    assert report.positive_frequency_phase_peak_certified
    assert report.phase_markov_bounds[2].contains(0)
    assert (
        pickle.dumps((graph.graph, dict(graph.nodes(data=True)), dict(graph.edges)))
        == before
    )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-mediated-response.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    path = tmp_path / "remote-response.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text()) == payload


def _pair_graph(*, epi=(1, 0), phase=(0, 0), capacity=(1, 1)):
    graph = nx.path_graph(2)
    for i in graph:
        graph.nodes[i].update(EPI=epi[i], theta=phase[i], nu_f=capacity[i])
    return graph


@pytest.mark.parametrize("capacity", [(1, 1), (1, 3), (0, 2)])
def test_reversible_pair_reduction_preserves_both_rows_and_weighted_invariants(
    capacity,
):
    graph = _pair_graph(
        epi=(Q(1, 4), Q(-1, 4)), phase=(Q(1, 4), Q(-1, 4)), capacity=capacity
    )
    report = assess_sine_pair_pulse(graph, reference_model=_model(epi_weight=0))
    assert report.form_contrast == Q(1, 2)
    assert report.phase_contrast == Q(-1, 2)
    assert report.capacity_sum == sum(capacity)
    assert report.initial_continuous_loss == 0
    with mp.workdps(100):
        u, delta, sigma = mp.mpf("0.5"), mp.mpf("-0.5"), sum(capacity)
        source = mp.sin(delta) / mp.pi
        form_rates = [capacity[0] * source, -capacity[1] * source]
        phase_rates = [capacity[0] * u / mp.pi, -capacity[1] * u / mp.pi]
        for i in range(2):
            _contains(report.comparison.form_rates[i], form_rates[i])
            _contains(report.comparison.phase_rates[i], phase_rates[i])
        u_rate = form_rates[0] - form_rates[1]
        delta_rate = phase_rates[1] - phase_rates[0]
        _contains(report.relative_form_rate_bounds, u_rate)
        _contains(report.relative_phase_rate_bounds, delta_rate)
        assert abs(capacity[1] * form_rates[0] + capacity[0] * form_rates[1]) == 0
        assert abs(capacity[1] * phase_rates[0] + capacity[0] * phase_rates[1]) == 0
        assert abs(u * u_rate + mp.sin(delta) * delta_rate) < mp.mpf("1e-90")
        acceleration = -sigma * u_rate / mp.pi
        assert mp.almosteq(acceleration, -((sigma / mp.pi) ** 2) * mp.sin(delta))


@pytest.mark.parametrize("form_gap,beta", [(Q(1), 1), (Q(7, 4), 1), (Q(2), 4)])
def test_nonlinear_pair_period_encloses_independent_elliptic_integral(form_gap, beta):
    model = _model(beta, epi_weight=0)
    report = assess_sine_pair_pulse(
        _pair_graph(epi=(form_gap, 0)), reference_model=model
    )
    assert report.status == "libration_certified"
    assert report.nonlinear_periodic_exchange_certified
    assert not report.nonstationary_recurrence_excluded
    assert report.exact_energy == form_gap**2 / 2
    with mp.workdps(100):
        energy = _mp(form_gap) ** 2 / 2
        parameter = energy / (2 * beta)
        exact_period = 2 * mp.pi * mp.sqrt(beta) * mp.ellipk(parameter)
        small_period = mp.pi**2 * mp.sqrt(beta)
        _contains(report.small_amplitude_period_bounds, small_period)
        _contains(report.period_bounds, exact_period)
        assert small_period < exact_period <= small_period / mp.sqrt(1 - parameter)
        assert report.period_unavailable_reasons == ()
        faster = assess_sine_pair_pulse(
            _pair_graph(epi=(form_gap, 0), capacity=(3, 3)), reference_model=model
        )
        frozen_endpoint = assess_sine_pair_pulse(
            _pair_graph(epi=(form_gap, 0), capacity=(0, 2)), reference_model=model
        )
        _contains(faster.period_bounds, exact_period / 3)
        _contains(frozen_endpoint.period_bounds, exact_period)


def test_pair_energy_boundaries_do_not_become_false_periodic_certificates():
    model = _model(epi_weight=0)
    equilibrium = assess_sine_pair_pulse(_pair_graph(epi=(0, 0)), reference_model=model)
    separatrix = assess_sine_pair_pulse(_pair_graph(epi=(2, 0)), reference_model=model)
    rotation = assess_sine_pair_pulse(_pair_graph(epi=(3, 0)), reference_model=model)
    frozen = assess_sine_pair_pulse(_pair_graph(capacity=(0, 0)), reference_model=model)
    ambiguous = assess_sine_pair_pulse(
        _pair_graph(epi=(0, 0), phase=(0, Q(1, 2**200))), reference_model=model
    )
    assert equilibrium.status == "equilibrium"
    assert separatrix.status == "separatrix_out_of_scope"
    assert rotation.status == "rotation_out_of_scope"
    assert frozen.status == "frozen"
    assert ambiguous.status == "energy_classification_unresolved"
    assert ambiguous.normalized_energy_bounds.contains(0)
    for report in (equilibrium, separatrix, rotation, frozen, ambiguous):
        assert not report.nonlinear_periodic_exchange_certified
        assert report.period_bounds is None
        assert report.period_unavailable_reasons
    near_separatrix = assess_sine_pair_pulse(
        _pair_graph(epi=(2, Q(1, 2**200))), reference_model=model
    )
    assert near_separatrix.exact_energy < near_separatrix.separatrix_energy
    assert near_separatrix.nonlinear_periodic_exchange_certified
    assert near_separatrix.period_bounds is None
    assert near_separatrix.period_unavailable_reasons == (
        "positive_period_bound_denominator_not_resolved",
    )


def test_zero_instantaneous_loss_or_tiny_damping_does_not_supply_permanent_motion():
    graph = _pair_graph(epi=(0, 0), phase=(0, Q(1, 2)))
    for model in (_model(), _model(epi_weight=Q(1, 2**200))):
        report = assess_sine_pair_pulse(graph, reference_model=model)
        assert report.initial_continuous_loss == 0
        assert report.status == "dissipative_recurrence_excluded"
        assert report.nonstationary_recurrence_excluded
        assert not report.nonlinear_periodic_exchange_certified
        assert report.period_bounds is None
        assert report.relative_form_rate_bounds.lo > 0
    with pytest.raises(ValueError, match="positive epi_weight"):
        _mode(model=_model(epi_weight=0))


def test_exact_tiny_positive_capacity_retains_a_finite_reversible_pair_period():
    tiny = Q(1, 2**200)
    report = assess_sine_pair_pulse(
        _pair_graph(capacity=(tiny, tiny)), reference_model=_model(epi_weight=0)
    )
    assert report.capacity_sum == 2 * tiny
    assert report.status == "libration_certified"
    assert report.period_bounds.lo > 0
    with mp.workdps(110):
        _contains(
            report.period_bounds, 2 * mp.pi * mp.ellipk(mp.mpf(1) / 4) / _mp(tiny)
        )


def test_pair_pulse_full_support_admission_detachment_and_exact_export(tmp_path):
    model = _model(epi_weight=0)
    graph = _pair_graph(epi=(Q(7, 4), 0), capacity=(1, 3))
    before = pickle.dumps(
        (graph.graph, dict(graph.nodes(data=True)), dict(graph.edges))
    )
    report = assess_sine_pair_pulse(graph, reference_model=model)
    assert (
        pickle.dumps((graph.graph, dict(graph.nodes(data=True)), dict(graph.edges)))
        == before
    )
    for unsupported in (nx.path_graph(3), nx.cycle_graph(3)):
        for node in unsupported:
            unsupported.nodes[node].update(EPI=0, theta=0, nu_f=1)
        with pytest.raises(ValueError):
            assess_sine_pair_pulse(unsupported, reference_model=model)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-pair-pulse.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    path = tmp_path / "pair-pulse.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text()) == payload


def _path_graph(*, epi=(1, 0, 0), phase=(0, 0, 0), capacity=1):
    graph = nx.path_graph(3)
    for i in graph:
        graph.nodes[i].update(EPI=epi[i], theta=phase[i], nu_f=capacity)
    return graph


def _q_product(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _mp_matrix(values):
    return mp.matrix([[_mp(value) for value in row] for row in values])


def test_nonlinear_path_edge_loss_is_reversible_transfer_not_total_dissipation():
    model = _model(Q(3, 2), epi_weight=0)
    graph = _path_graph(phase=(0, 0, Q(1, 2)), capacity=Q(3, 2))
    report = assess_sine_path_memory(graph, reference_model=model, mediator=1)
    reverse = assess_sine_path_memory(
        _path_graph(phase=(0, 0, Q(-1, 2)), capacity=Q(3, 2)),
        reference_model=model,
        mediator=1,
    )
    assert report.comparison.continuous_loss == reverse.comparison.continuous_loss == 0
    with mp.workdps(100):
        beta, nu = mp.mpf("1.5"), mp.mpf("1.5")
        left_rate = -nu * mp.sin(mp.mpf("0.5")) / (2 * mp.pi)
        _contains(report.nonlinear_transfer_bounds, left_rate)
        _contains(reverse.nonlinear_transfer_bounds, -left_rate)
        for i, energy in enumerate((mp.mpf("0.5"), beta * (1 - mp.cos(mp.mpf("0.5"))))):
            _contains(report.nonlinear_edge_storage_bounds[i], energy)
            _contains(reverse.nonlinear_edge_storage_bounds[i], energy)
            _contains(
                report.nonlinear_edge_rate_bounds[i], left_rate * (1 if i == 0 else -1)
            )
            assert report.nonlinear_edge_rate_residual_bounds[i].contains(0)
        _contains(report.comparison.storage_rate, 0)


def test_conservative_path_generator_is_full_consensus_derivative_in_declared_clock():
    beta, nu = Q(3, 2), Q(3, 2)
    graph = _path_graph(
        epi=(Q(1, 4), Q(-1, 2), Q(3, 4)),
        phase=(Q(1, 8), Q(1, 4), Q(-1, 8)),
        capacity=nu,
    )
    report = assess_sine_path_memory(
        graph,
        reference_model=_model(beta, epi_weight=0),
        mediator=1,
        consensus_form=Q(1, 4),
        consensus_phase=Q(1, 8),
    )
    laplacian = tuple(
        tuple(
            Q(graph.degree[i] if i == j else -int(graph.has_edge(i, j))) for j in graph
        )
        for i in graph
    )
    transport = tuple(
        tuple(nu * value / graph.degree[i] for value in row)
        for i, row in enumerate(laplacian)
    )
    zero = (Q(0),) * 3
    expected = tuple(
        zero + tuple(-value for value in row) for row in transport
    ) + tuple(tuple(value / beta for value in row) + zero for row in transport)
    assert report.coordinate_memory.generator == expected
    storage_metric = tuple(
        tuple(
            (
                laplacian[i][j]
                if i < 3 and j < 3
                else beta * laplacian[i - 3][j - 3] if i >= 3 and j >= 3 else Q(0)
            )
            for j in range(6)
        )
        for i in range(6)
    )
    product = _q_product(storage_metric, expected)
    assert all(product[i][j] + product[j][i] == 0 for i in range(6) for j in range(6))
    with mp.workdps(100):
        reference = mp.matrix(
            [_mp(report.consensus_form)] * 3 + [_mp(report.consensus_phase)] * 3
        )

        def tau_field(state):
            form = [
                sum(mp.sin(state[3 + j] - state[3 + i]) for j in graph[i])
                * _mp(nu)
                / graph.degree[i]
                for i in graph
            ]
            phase = [
                sum(state[i] - state[j] for j in graph[i])
                * _mp(nu / beta)
                / graph.degree[i]
                for i in graph
            ]
            return form + phase

        for column in range(6):
            direction = mp.matrix([int(i == column) for i in range(6)])
            for row in range(6):
                derivative = mp.diff(
                    lambda t: tau_field(reference + t * direction)[row], 0
                )
                assert mp.almosteq(
                    derivative, _mp(expected[row][column]), abs_eps=mp.mpf("1e-90")
                )
        _contains(report.clock_rate_bounds, 1 / mp.pi)
        _contains(
            report.full_tangent_period_bounds,
            2 * mp.pi**2 * mp.sqrt(_mp(beta)) / _mp(nu),
        )


def test_conservative_hidden_memory_keeps_initial_state_and_prevents_instantaneous_closure():
    graph = _path_graph(epi=(0, Q(1, 2), 0), phase=(0, Q(1, 4), 0))
    report = assess_sine_path_memory(
        graph, reference_model=_model(epi_weight=0), mediator=1
    )
    memory = report.coordinate_memory
    assert memory.visible_indices == (0, 2, 3, 5)
    assert memory.hidden_indices == (1, 4)
    assert report.linear_observation.dimension == 6
    assert report.linear_observation.extra_coordinates == 2
    assert report.hidden_initial_state == (Q(1, 2), Q(1, 4))
    d_squared = _q_product(memory.hidden_generator, memory.hidden_generator)
    assert d_squared == ((Q(-1), Q(0)), (Q(0), Q(-1)))
    assert report.hidden_frequency_squared == 1
    bc = _q_product(memory.hidden_to_visible, memory.visible_to_hidden)
    bdc = _q_product(
        _q_product(memory.hidden_to_visible, memory.hidden_generator),
        memory.visible_to_hidden,
    )
    assert memory.kernel_at_zero == bc
    assert report.kernel_derivative_at_zero == bdc
    initial_column = tuple((value,) for value in report.hidden_initial_state)
    assert report.hidden_initial_forcing == tuple(
        row[0] for row in _q_product(memory.hidden_to_visible, initial_column)
    )
    assert report.hidden_initial_forcing_derivative == tuple(
        row[0]
        for row in _q_product(
            _q_product(memory.hidden_to_visible, memory.hidden_generator),
            initial_column,
        )
    )
    assert any(report.hidden_initial_forcing)
    empty = assess_sine_path_memory(
        _path_graph(epi=(0, 0, 0)), reference_model=_model(epi_weight=0), mediator=1
    )
    assert tuple(
        report.initial_tangent_state[i] for i in memory.visible_indices
    ) == tuple(empty.initial_tangent_state[i] for i in memory.visible_indices)
    assert all(value == 0 for value in empty.hidden_initial_forcing)
    assert (
        not empty.tangent_motion_nonstationary and report.tangent_motion_nonstationary
    )
    # Equal endpoint observations have different instantaneous rates; replacing
    # the hidden source by damping would change the initial-value problem.


def test_path_tangent_edge_energy_returns_after_reversible_transfer():
    report = assess_sine_path_memory(
        _path_graph(), reference_model=_model(epi_weight=0), mediator=1
    )
    with mp.workdps(100):
        generator = _mp_matrix(report.coordinate_memory.generator)
        initial = mp.matrix([_mp(value) for value in report.initial_tangent_state])
        for tau in (mp.pi / 2, mp.pi, 2 * mp.pi):
            state = mp.expm(tau * generator) * initial
            left = ((state[0] - state[1]) ** 2 + (state[3] - state[4]) ** 2) / 2
            right = ((state[1] - state[2]) ** 2 + (state[4] - state[5]) ** 2) / 2
            assert mp.almosteq(left, (1 + mp.cos(tau)) / 4, abs_eps=mp.mpf("1e-90"))
            assert mp.almosteq(right, (1 - mp.cos(tau)) / 4, abs_eps=mp.mpf("1e-90"))
            assert mp.almosteq(left + right, mp.mpf("0.5"))
        assert mp.norm(state - initial) < mp.mpf("1e-90")
        _contains(report.full_tangent_period_bounds, 2 * mp.pi**2)
    # This finite matrix identity concerns quadratic tangent storage, not the
    # nonlinear cosine energy or a finite-amplitude nonlinear return claim.


def test_connected_conservative_cycle_does_not_have_one_common_tangent_period():
    graph = nx.cycle_graph(5)
    with mp.workdps(100):
        spatial = _laplacian(graph) / 2
        eigenvalues = mp.eigsy(spatial, eigvals_only=True)
        low, high = (5 - mp.sqrt(5)) / 4, (5 + mp.sqrt(5)) / 4
        assert abs(eigenvalues[1] - low) < mp.mpf("1e-90")
        assert abs(eigenvalues[3] - high) < mp.mpf("1e-90")
        ratio = high / low
        assert abs(ratio - (3 + mp.sqrt(5)) / 2) < mp.mpf("1e-90")
    # The ideal ratio solves r^2-3r+1=0. Its integer discriminant is not
    # a square, proving irrationality independently of the numerical spectrum.
    discriminant = 3**2 - 4
    assert isqrt(discriminant) ** 2 != discriminant
    # The ratio survives all common capacity, beta and clock rescalings;
    # observing both modes therefore need not produce a periodic tangent signal.


def test_path_memory_admission_origins_exact_capacity_and_detached_export(tmp_path):
    graph = _path_graph(epi=(Q(1, 4), Q(-1, 2), Q(3, 4)), phase=(0, Q(1, 4), 0))
    model = _model(epi_weight=0)
    before = pickle.dumps(
        (graph.graph, dict(graph.nodes(data=True)), dict(graph.edges))
    )
    report = assess_sine_path_memory(graph, reference_model=model, mediator=1)
    shifted = assess_sine_path_memory(
        graph, reference_model=model, mediator=1, consensus_form=3, consensus_phase=-2
    )
    generator = report.coordinate_memory.generator
    assert _q_product(generator, tuple((v,) for v in report.initial_tangent_state)) == (
        _q_product(generator, tuple((v,) for v in shifted.initial_tangent_state))
    )
    assert (
        pickle.dumps((graph.graph, dict(graph.nodes(data=True)), dict(graph.edges)))
        == before
    )
    with pytest.raises(ValueError, match="zero EPI weight"):
        assess_sine_path_memory(graph, reference_model=_model(), mediator=1)
    for changes in ({"mediator": 0}, {"mediator": 99}, {"consensus_phase": True}):
        arguments = dict(reference_model=model, mediator=1)
        arguments.update(changes)
        with pytest.raises((ValueError, TypeError)):
            assess_sine_path_memory(graph, **arguments)
    unequal = graph.copy()
    unequal.nodes[1]["nu_f"] = 2
    for unsupported in (unequal, _path_graph(capacity=0), _pair_graph()):
        with pytest.raises(ValueError):
            assess_sine_path_memory(unsupported, reference_model=model, mediator=1)
    tiny = Q(1, 2**200)
    exact = assess_sine_path_memory(
        _path_graph(capacity=tiny), reference_model=model, mediator=1
    )
    assert exact.hidden_frequency_squared == tiny**2
    assert exact.full_tangent_period_bounds.lo > 0
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-path-memory.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    path = tmp_path / "conservative-memory.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text()) == payload


def _recurrence(graph, **kwargs):
    arguments = dict(
        reference_model=_model(epi_weight=0),
        energy_ceiling=3,
        form_mean_bounds=(-2, 2),
    )
    arguments.update(kwargs)
    return assess_sine_recurrence(graph, **arguments)


def test_nonlinear_recurrence_invariants_and_volume_use_the_complete_heterogeneous_field():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    forms = (Q(1, 2), Q(-1, 4), Q(3, 4), Q(0))
    phases = (Q(1, 8), Q(-1, 4), Q(3, 8), Q(-1, 8))
    capacity = (Q(1), Q(2), Q(3, 2), Q(3, 8))
    beta = Q(3, 2)
    for node in graph:
        graph.nodes[node].update(
            EPI=forms[node], theta=phases[node], nu_f=capacity[node]
        )
    report = _recurrence(graph, reference_model=_model(beta, epi_weight=0))
    weights = tuple(Q(graph.degree[i]) / capacity[i] for i in graph)
    mass = sum(weights)
    mean = sum(weight * form for weight, form in zip(weights, forms)) / mass
    assert report.form_mean_weights == weights
    assert report.normalized_mean_weights == tuple(weight / mass for weight in weights)
    assert report.weighted_form_mean == mean
    assert report.divergence == 0
    assert report.snapshot_family_membership == "inside"
    assert report.snapshot_motion_status == "nonstationary"
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
    assert report.almost_everywhere_recurrence_certified
    assert report.finite_positive_family_volume_certified
    with mp.workdps(100):
        laplacian = _laplacian(graph)
        state = mp.matrix([_mp(value) for value in forms + phases])

        def full_field(value):
            q = laplacian * value[:4, 0]
            current = [
                sum(mp.sin(value[4 + j] - value[4 + i]) for j in graph[i])
                for i in graph
            ]
            mobility = [_mp(capacity[i]) / graph.degree[i] for i in graph]
            return mp.matrix(
                [mobility[i] * current[i] / mp.pi for i in graph]
                + [mobility[i] * q[i] / (_mp(beta) * mp.pi) for i in graph]
            )

        field = full_field(state)
        q = laplacian * state[:4, 0]
        phase_gradient = mp.matrix(
            [
                -_mp(beta) * sum(mp.sin(state[4 + j] - state[4 + i]) for j in graph[i])
                for i in graph
            ]
        )
        storage_rate = (q.T * field[:4, 0])[0] + (phase_gradient.T * field[4:, 0])[0]
        mean_rate = sum(_mp(weights[i]) * field[i] for i in graph) / _mp(mass)
        divergence = sum(
            mp.diff(
                lambda displacement: full_field(
                    state + mp.matrix([int(j == i) for j in range(8)]) * displacement
                )[i],
                0,
            )
            for i in range(8)
        )
        assert abs(storage_rate) < mp.mpf("1e-90")
        assert abs(mean_rate) < mp.mpf("1e-90")
        assert divergence == 0
        _contains(report.weighted_mean_rate_residual_bounds, mean_rate)
        _contains(report.comparison.storage_rate, storage_rate)
        for i in graph:
            _contains(report.comparison.form_rates[i], field[i])
            _contains(report.comparison.phase_rates[i], field[4 + i])


def test_declared_recurrence_family_has_a_compact_form_box_without_bounding_phase_lifts():
    graph = _path_graph(epi=(Q(-1, 2), 0, Q(3, 4)), phase=(0, Q(1, 8), 0))
    graph.nodes[0]["nu_f"] = Q(1, 2)
    graph.nodes[2]["nu_f"] = Q(3)
    report = _recurrence(graph, energy_ceiling=2, form_mean_bounds=(-1, 2))
    assert report.path_length_upper_bound == len(graph) - 1
    with mp.workdps(100):
        radius = mp.sqrt(2 * report.energy_ceiling * (len(graph) - 1))
        _contains(report.form_radius_bounds, radius)
        assert _mp(report.form_coordinate_bounds.lo) <= -1 - radius
        assert _mp(report.form_coordinate_bounds.hi) >= 2 + radius
        for value in report.comparison.epi:
            assert abs(_mp(value - report.weighted_form_mean)) <= radius
            assert report.form_coordinate_bounds.contains(value)
    translated = graph.copy()
    for node in translated:
        translated.nodes[node]["EPI"] += 7
        translated.nodes[node]["theta"] += Q(10000, 3)
    shifted = _recurrence(translated, energy_ceiling=2, form_mean_bounds=(6, 9))
    assert shifted.weighted_form_mean == report.weighted_form_mean + 7
    assert shifted.comparison.storage == report.comparison.storage
    assert shifted.comparison.form_rates == report.comparison.form_rates
    assert shifted.comparison.phase_rates == report.comparison.phase_rates
    assert shifted.form_coordinate_bounds.lo == report.form_coordinate_bounds.lo + 7
    assert shifted.form_coordinate_bounds.hi == report.form_coordinate_bounds.hi + 7


def test_family_membership_does_not_resolve_an_exact_nonrecurrent_separatrix():
    graph = _pair_graph(epi=(2, 0))
    report = _recurrence(graph, energy_ceiling=2, form_mean_bounds=(0, 2))
    assert report.captured_energy_exact == 2
    assert report.snapshot_family_membership == "inside"
    assert report.snapshot_motion_status == "nonstationary"
    assert report.almost_everywhere_recurrence_certified
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
    pulse = assess_sine_pair_pulse(graph, reference_model=_model(epi_weight=0))
    assert pulse.status == "separatrix_out_of_scope"
    # This analytic separatrix starts at the exact captured point. Its form
    # contrast decreases strictly for every t>0 and tends to zero, so that
    # point is a nonrecurrent exception despite belonging to the family.
    with mp.workdps(100):
        omega = 2 / mp.pi

        def form_gap(time):
            return 2 / mp.cosh(omega * time)

        def phase_gap(time):
            return -2 * mp.asin(mp.tanh(omega * time))

        for time in (mp.mpf(0), mp.mpf("0.25"), mp.mpf(1)):
            u, delta = form_gap(time), phase_gap(time)
            assert abs(mp.diff(form_gap, time) - 2 * mp.sin(delta) / mp.pi) < mp.mpf(
                "1e-90"
            )
            assert abs(mp.diff(phase_gap, time) + 2 * u / mp.pi) < mp.mpf("1e-90")
            assert abs(u**2 / 2 + 1 - mp.cos(delta) - 2) < mp.mpf("1e-90")
            if time > 0:
                assert form_gap(time) < 2
                assert mp.diff(form_gap, time) < 0


def test_recurrence_membership_and_motion_ambiguities_remain_separate():
    model = _model(epi_weight=0)
    stationary = _recurrence(_pair_graph(epi=(0, 0)))
    assert stationary.snapshot_motion_status == "stationary"
    assert stationary.individual_recurrence_status == "trivial_stationary_recurrence"
    for graph, reason in (
        (_pair_graph(epi=(3, 0)), "exact_energy_exceeds_declared_ceiling"),
        (_pair_graph(epi=(4, 4)), "weighted_mean_outside_declared_slab"),
    ):
        outside = _recurrence(graph, energy_ceiling=2)
        assert outside.snapshot_family_membership == "outside"
        assert outside.snapshot_membership_reasons == (reason,)
        assert outside.almost_everywhere_recurrence_certified
    tiny = Q(1, 2**200)
    boundary = _recurrence(
        _pair_graph(epi=(2, 0), phase=(0, tiny)),
        reference_model=model,
        energy_ceiling=2,
    )
    assert boundary.captured_energy_exact is None
    assert boundary.snapshot_family_membership == "unresolved"
    assert boundary.snapshot_membership_reasons == (
        "energy_ceiling_membership_not_resolved",
    )
    assert boundary.snapshot_motion_status == "nonstationary"
    phase_only = _recurrence(_pair_graph(epi=(0, 0), phase=(0, tiny)))
    assert phase_only.snapshot_motion_status == "unresolved"
    assert phase_only.individual_recurrence_status == "unavailable_for_chosen_state"
    tiny_gradient = _recurrence(_pair_graph(epi=(tiny, 0)))
    assert tiny_gradient.snapshot_motion_status == "nonstationary"
    assert tiny_gradient.individual_recurrence_status == "unavailable_for_chosen_state"


@pytest.mark.parametrize(
    "changes",
    [
        {"energy_ceiling": 0},
        {"energy_ceiling": -1},
        {"energy_ceiling": True},
        {"energy_ceiling": float("inf")},
        {"form_mean_bounds": (0, 0)},
        {"form_mean_bounds": (1, -1)},
        {"form_mean_bounds": (False, 1)},
        {"form_mean_bounds": (0,)},
        {"form_mean_bounds": {0, 1}},
    ],
)
def test_recurrence_family_rejects_invalid_or_zero_volume_declarations(changes):
    with pytest.raises((ValueError, TypeError)):
        _recurrence(_pair_graph(), **changes)


def test_recurrence_scope_exact_tiny_capacity_and_detached_sdk_export(tmp_path):
    graph = _pair_graph(capacity=(Q(1, 2**200), 1))
    before = pickle.dumps(
        (graph.graph, dict(graph.nodes(data=True)), dict(graph.edges))
    )
    report = _recurrence(graph)
    assert report.form_mean_weights == (Q(2**200), Q(1))
    assert report.weighted_form_mean == Q(2**200, 2**200 + 1)
    assert report.snapshot_motion_status == "nonstationary"
    assert report.almost_everywhere_recurrence_certified
    assert (
        pickle.dumps((graph.graph, dict(graph.nodes(data=True)), dict(graph.edges)))
        == before
    )
    for capacity in ((0, 1), (-1, 1)):
        with pytest.raises(ValueError):
            _recurrence(_pair_graph(capacity=capacity))
    with pytest.raises(ValueError, match="zero EPI weight"):
        _recurrence(_pair_graph(), reference_model=_model(epi_weight=Q(1, 2**200)))
    disconnected = _pair_graph()
    disconnected.remove_edge(0, 1)
    weighted = _pair_graph()
    weighted[0][1]["weight"] = 2
    for unsupported in (disconnected, weighted, nx.empty_graph(1)):
        with pytest.raises(ValueError):
            _recurrence(unsupported)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-recurrence.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    path = tmp_path / "nonlinear-recurrence-family.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text()) == payload
