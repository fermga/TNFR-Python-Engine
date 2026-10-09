"""Static nonlinear port memory and response-free fidelity controls."""

import subprocess
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tnfr.physics import _sine_class_collective_interface as owner


@pytest.fixture(scope="module", autouse=True)
def no_response_coefficient_or_worker():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("collective interface controls cannot run a scientific producer")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (_sine_flow, ("_full_sine_field",)),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                relational_sine_class_cubic_response,
                (
                    "_class_cubic_coefficients",
                    "_time_coefficients",
                    "_coefficient_segment",
                ),
            ),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (
                relational_sine_class_comparison_readout,
                ("bound_sine_class_comparison_readout",),
            ),
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


def _graph():
    components = tuple(
        tuple((9 * c + j, 9 * c + (j + 1) % 9) for j in range(9)) for c in range(3)
    )
    edges = sum(components, ()) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))

    def matrix(selected):
        result = [[Q(0)] * 27 for _ in range(27)]
        for i, j in selected:
            result[i][i] += Q(1, degrees[i])
            result[i][j] -= Q(1, degrees[i])
            result[j][j] += Q(1, degrees[j])
            result[j][i] -= Q(1, degrees[j])
        return tuple(map(tuple, result))

    return (
        edges,
        degrees,
        matrix(edges),
        tuple(matrix(e) for e in components + (edges[-2:],)),
    )


def _mv(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Q(0))


@pytest.fixture(scope="module", params=(1, 2))
def interface(request):
    return owner._derive_collective_interface(request.param)


def test_original_coordinate_partition_matches_independent_fine_graph(interface):
    _, degrees, a, components = _graph()
    ports = (4, 13, 22)
    hidden = tuple(i for i in range(27) if i not in ports)
    assert interface.port_indices == (4, 13, 22, 31, 40, 49)
    assert interface.hidden_indices == hidden + tuple(i + 27 for i in hidden)
    assert len(interface.hidden_indices) == 48
    assert interface.parameter_bounds.degrees == degrees
    for actual, expected in zip(
        (interface.spatial_partition,) + interface.phase_component_partitions,
        (a,) + components,
    ):
        assert actual.generator == expected
        assert actual.visible_indices == ports and actual.hidden_indices == hidden
        assert actual.visible_generator == tuple(
            tuple(expected[i][j] for j in ports) for i in ports
        )
        assert actual.hidden_generator == tuple(
            tuple(expected[i][j] for j in hidden) for i in hidden
        )
        assert actual.hidden_to_visible == tuple(
            tuple(expected[i][j] for j in hidden) for i in ports
        )
        assert actual.visible_to_hidden == tuple(
            tuple(expected[i][j] for j in ports) for i in hidden
        )
    assert all(sum(a[i]) == 0 for i in range(27))


def test_hidden_paths_and_reflection_supply_exact_modal_structure(interface):
    expected = tuple(
        tuple(Q(int(i == j)) - Q(int(abs(i - j) == 1), 2) for j in range(8))
        for i in range(8)
    )
    assert interface.normalized_grounded_path_matrix == expected
    reflection = interface.reflection_indices
    assert tuple(reflection[i] for i in reflection) == tuple(range(54))
    assert all(reflection[i] == i for i in interface.port_indices)
    for component, nodes in enumerate(interface.hidden_path_node_orders):
        assert nodes == tuple(9 * component + j for j in (5, 6, 7, 8, 0, 1, 2, 3))
        assert tuple(reflection[node] for node in nodes) == nodes[::-1]
    # An independent sine eigenbasis checks the declared path spectrum;
    # no hidden or full dynamical matrix exponential is evaluated.
    matrix = np.array(expected, dtype=float)
    endpoint_projections = []
    for mode in range(1, 9):
        vector = np.sin(np.arange(1, 9) * mode * np.pi / 9)
        np.testing.assert_allclose(
            matrix @ vector,
            (1 - np.cos(mode * np.pi / 9)) * vector,
            rtol=1e-13,
            atol=1e-14,
        )
        normalized = np.sqrt(2 / 9) * vector
        endpoint_sum = normalized[0] + normalized[-1]
        endpoint_projections.append(endpoint_sum**2)
        if mode % 2 == 0:
            assert abs(endpoint_sum) < 1e-14
        else:
            for degree in (3, 4):
                assert np.isclose(
                    endpoint_sum**2 / (2 * degree),
                    4 * np.sin(mode * np.pi / 9) ** 2 / (9 * degree),
                )
    assert np.isclose(sum(endpoint_projections), 2)
    for degree in (3, 4):
        assert np.isclose(sum(endpoint_projections) / (2 * degree), 1 / degree)


def test_two_channel_partition_retains_original_gamma_clock_factors(interface):
    edges, degrees, _, _ = _graph()
    gamma = Q(1, 7)
    cosines = (Q(3, 4), Q(1, 5), Q(3, 4), Q(1))
    x = tuple(Q((i * 7) % 13 - 6, 100) for i in range(27))
    y = tuple(Q((i * 5) % 17 - 8, 100) for i in range(27))
    direct_x, direct_y = [Q(0)] * 27, [Q(0)] * 27
    for index, (i, j) in enumerate(edges):
        cosine = cosines[min(index // 9, 3)]
        form_current = x[j] - x[i] + gamma * cosine * (y[j] - y[i])
        phase_current = -gamma * (x[j] - x[i])
        for row, current in ((direct_x, form_current), (direct_y, phase_current)):
            row[i] += current / degrees[i]
            row[j] -= current / degrees[j]

    def block_rates(name, inputs):
        a = getattr(interface.spatial_partition, name)
        phase_blocks = tuple(
            getattr(part, name) for part in interface.phase_component_partitions
        )
        c = tuple(
            tuple(
                sum(weight * part[i][j] for weight, part in zip(cosines, phase_blocks))
                for j in range(len(a[0]))
            )
            for i in range(len(a))
        )
        au = _mv(a, tuple(x[i] for i in inputs))
        cy = _mv(c, tuple(y[i] for i in inputs))
        return tuple(-u - gamma * v for u, v in zip(au, cy)), tuple(
            gamma * u for u in au
        )

    visible = interface.spatial_partition.visible_indices
    hidden = interface.spatial_partition.hidden_indices
    for outputs, first, second in (
        (
            visible,
            block_rates("visible_generator", visible),
            block_rates("hidden_to_visible", hidden),
        ),
        (
            hidden,
            block_rates("visible_to_hidden", visible),
            block_rates("hidden_generator", hidden),
        ),
    ):
        assert tuple(a + b for a, b in zip(first[0], second[0])) == tuple(
            direct_x[i] for i in outputs
        )
        assert tuple(a + b for a, b in zip(first[1], second[1])) == tuple(
            direct_y[i] for i in outputs
        )
    assert _dot(degrees, direct_x) == _dot(degrees, direct_y) == 0


def test_original_edge_tensor_factors_enclose_direct_derivatives(interface):
    mp = mpmath.mp.clone()
    mp.dps = 70

    def real(q):
        return mp.mpf(q.numerator) / q.denominator

    edges, degrees, _, _ = _graph()
    classes = (1, interface.mediator_class, 1)
    gamma = 1 / (1023 * mp.pi)
    for index, (i, j) in enumerate(edges):
        gap = (
            2
            * mp.pi
            * (
                Q(classes[j // 9] * (j % 9 - 4), 9)
                - Q(classes[i // 9] * (i % 9 - 4), 9)
            )
        )
        for actual, expected in (
            (interface.quadratic_edge_scalar_bounds[index], -gamma * mp.sin(gap) / 2),
            (interface.cubic_edge_scalar_bounds[index], -gamma * mp.cos(gap) / 6),
        ):
            assert real(actual.lo) <= expected <= real(actual.hi)
        assert interface.edge_phase_difference_rows[index] == tuple(
            Q(int(n == j) - int(n == i)) for n in range(27)
        )
        current = interface.normalized_edge_current_columns[index]
        assert current == tuple(
            Q(int(n == i) - int(n == j), degrees[n]) for n in range(27)
        )
        assert _dot(degrees, current) == 0


def test_nominal_hidden_quadratic_term_can_reenter_central_cubic_row(interface):
    # Exact algebraic edge weights with the required target parity test the
    # structural tensor identity, without evaluating finite time coefficients.
    sine = (Q(2, 3),) * 9 + (Q(4, 5),) * 9 + (Q(2, 3),) * 9 + (Q(0),) * 2
    even = tuple(Q((i % 9 - 4) ** 2, 20) for i in range(27))
    odd = tuple(Q(i % 9 - 4, 7) for i in range(27))

    def quadratic(left, right):
        out = [Q(0)] * 27
        for row, column, scalar in zip(
            interface.edge_phase_difference_rows,
            interface.normalized_edge_current_columns,
            sine,
        ):
            force = -scalar * _dot(row, left) * _dot(row, right) / 2
            out = [
                value + coefficient * force for value, coefficient in zip(out, column)
            ]
        return tuple(out)

    second = quadratic(even, even)
    reflected = interface.reflection_indices[:27]
    assert any(second)
    assert all(second[node] == 0 for node in interface.port_nodes)
    assert tuple(second[i] for i in reflected) == tuple(-value for value in second)
    coupling = tuple(2 * value for value in quadratic(even, odd))
    assert tuple(coupling[i] for i in reflected) == coupling
    assert all(coupling[node] != 0 for node in interface.port_nodes)
    assert interface.nominal_amplitude_reflection_eigenvalues == (1, -1, 1)
    assert (
        "nominal_central_quadratic_output_vanishes_but_hidden_quadratic_feedback_remains"
        in interface.scope
    )


def test_same_ports_with_different_hidden_state_have_different_future(interface):
    a = interface.spatial_partition.generator
    eps = Q(1, 10**32)
    perturbation = tuple(eps / 2 * (int(i == 3) - int(i == 2)) for i in range(27))
    assert all(perturbation[i] == 0 for i in interface.port_nodes)
    degrees = interface.parameter_bounds.degrees
    assert _dot(degrees, perturbation) == 0
    assert sum(d * value**2 for d, value in zip(degrees, perturbation)) == eps**2
    projected_rate = tuple(-_mv(a, perturbation)[i] for i in interface.port_nodes)
    assert projected_rate == (eps / 6, Q(0), Q(0))
    # Both sources lie in K with the same means and port readings. Their
    # different hidden states cannot be replaced by those retained values.
    assert projected_rate != (Q(0),) * 3


def test_boundary_pressure_and_simultaneous_port_work_keep_cross_terms(interface):
    _, degrees, a, _ = _graph()
    laplacian = tuple(
        tuple(degrees[i] * value for value in row) for i, row in enumerate(a)
    )
    expected = ((Q(3), Q(-1), Q(0)), (Q(-1), Q(4), Q(-1)), (Q(0), Q(-1), Q(3)))
    assert interface.port_jump_quadratic_matrix == expected
    assert interface.port_laplacian_observer == tuple(
        laplacian[i] for i in interface.port_nodes
    )
    assert tuple(
        sum(map(abs, row), Q(0)) for row in interface.port_laplacian_observer
    ) == (Q(6), Q(8), Q(6))
    before = tuple(Q((i * 3) % 17 - 8, 100) + Q(11, 7) for i in range(27))
    impulses = (Q(1, 10), Q(-2, 15), Q(3, 20))
    jump = tuple(
        sum(q * column[i] for q, column in zip(impulses, interface.port_jump_columns))
        for i in range(54)
    )
    assert jump[27:] == (Q(0),) * 27
    after = tuple(value + delta for value, delta in zip(before, jump[:27]))
    exact_work = (
        _dot(after, _mv(laplacian, after)) - _dot(before, _mv(laplacian, before))
    ) / 2
    pressure_work = _dot(impulses, _mv(interface.port_laplacian_observer, before))
    assert exact_work == pressure_work + _dot(impulses, _mv(expected, impulses)) / 2
    independent_self_terms = (
        sum(expected[i][i] * impulses[i] ** 2 for i in range(3)) / 2
    )
    assert _dot(impulses, _mv(expected, impulses)) / 2 != independent_self_terms


@pytest.mark.parametrize("bad", (True, np.int64(1), 0, 3, Q(1), 1.0))
def test_class_admission_precedes_target_parameters(bad, monkeypatch):
    monkeypatch.setattr(
        owner, "_cubic_parameters", lambda *_: pytest.fail("late admission")
    )
    with pytest.raises(ValueError):
        owner._derive_collective_interface(bad)


def _bound_arguments():
    return dict(
        total_input_variation=Q(7, 5000), horizon=Q(2), endpoint_radius=Q(1, 10**32)
    )


@pytest.mark.parametrize("key", tuple(_bound_arguments()))
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), float("nan"), float("inf"), Decimal("1e-400"))
)
def test_error_bound_admission_precedes_arithmetic(key, bad, monkeypatch):
    monkeypatch.setattr(
        owner, "_higher_amplitude_remainder", lambda *_: pytest.fail("late admission")
    )
    with pytest.raises((TypeError, ValueError)):
        owner._bound_collective_interface(**(_bound_arguments() | {key: bad}))


@pytest.mark.parametrize(
    "change",
    (
        {"total_input_variation": -1},
        {"total_input_variation": Q(7, 5000) + Q(1, 2**500)},
        {"horizon": -1},
        {"horizon": Q(2) + Q(1, 2**500)},
        {"endpoint_radius": -1},
    ),
)
def test_supported_error_domain_is_explicit(change):
    with pytest.raises(ValueError):
        owner._bound_collective_interface(**(_bound_arguments() | change))


@pytest.mark.parametrize(
    "amplitude,horizon,eps",
    (
        (Q(0), Q(2), Q(1, 10**32)),
        (Q(7, 5000), Q(0), Q(1, 10**32)),
        (Q(7, 5000), Q(2), Q(0)),
        (Q(1, 1000), Q(3, 2), Q(1, 10**100)),
    ),
)
def test_independent_fidelity_arithmetic_preserves_zero_and_subgrid_terms(
    amplitude, horizon, eps
):
    result = owner._bound_collective_interface(
        total_input_variation=amplitude, horizon=horizon, endpoint_radius=eps
    )
    g = Q(1, 3000)
    ell, d = 1 - 2 * g * horizon, 1 - 2 * g**2 * horizon**2
    nominal = 256 * g**6 * amplitude**5 * horizon / (d * (1 - 4 * g**2 * amplitude**2))
    initialization = 4 * g * horizon * eps * (g * amplitude / d + eps / ell) / (ell * d)
    assert result.nominal_fifth_order_error_upper_bound == nominal
    assert result.nonlinear_initialization_error_upper_bound == initialization
    assert result.form_error_upper_bound == nominal + initialization
    assert result.phase_error_upper_bound == 2 * g * horizon * (
        nominal + initialization
    )
    assert result.port_pressure_error_upper_bounds == tuple(
        value * (nominal + initialization) for value in (6, 8, 6)
    )
    if amplitude == 0 and eps > 0 and horizon > 0:
        assert initialization > 0
    if eps > 0 and amplitude > 0 and horizon > 0:
        assert 0 < initialization < Q(1, 2**128)


@pytest.fixture(scope="module")
def repeated():
    # Supplied rounded premise only; no saved coefficient or response is read.
    return owner._bound_repeated_collective_interface(
        base_cubic_lower=Q(-7014, 10**33), base_cubic_upper=Q(-7013, 10**33)
    )


def test_fixed_word_preserves_all_eight_initialization_terms_and_separate_memory(
    repeated,
):
    scale, g, eps, a = Q(7, 5), Q(1, 3000), Q(1, 10**32), Q(7, 10000)
    assert repeated.scaled_cubic_bounds == tuple(
        scale**3 * value for value in repeated.base_cubic_bounds
    )
    assert tuple(
        row.total_input_variation for row in repeated.per_history_fidelity
    ) == (0, a, a, 2 * a)
    fifth = 2 * sum(
        row.nominal_fifth_order_error_upper_bound
        for row in repeated.per_history_fidelity
    )
    initialization = 2 * sum(
        row.nonlinear_initialization_error_upper_bound
        for row in repeated.per_history_fidelity
    )
    assert (
        repeated.per_history_fidelity[0].nonlinear_initialization_error_upper_bound > 0
    )
    assert repeated.nominal_fifth_order_contrast_error_upper_bound == fifth
    assert (
        repeated.nonlinear_initialization_contrast_error_upper_bound == initialization
    )
    linear_source = 8 * eps / (1 - 4 * g)
    old = repeated.repeated
    assert (
        old.nonlinear_source_error_upper_bound
        == old.tangent_source_error_upper_bound
        == linear_source
    )
    assert old.nominal_contrast_bounds == (
        repeated.scaled_cubic_bounds[0] - fifth,
        repeated.scaled_cubic_bounds[1] + fifth,
    )
    assert repeated.full_response_via_interface_bounds == (
        old.true_contrast_bounds[0] - initialization,
        old.true_contrast_bounds[1] + initialization,
    )
    assert repeated.recorded_full_response_via_interface_bounds == (
        old.recorded_contrast_bounds[0] - initialization,
        old.recorded_contrast_bounds[1] + initialization,
    )
    assert (
        repeated.interface_separation_margin == old.separation_margin - initialization
    )
    assert repeated.conditional_repeated_interface_sufficient
    assert (
        old.all_work_within_allowances
        and old.nonlinear_return_sufficient
        and old.tangent_return_sufficient
    )


@pytest.mark.parametrize("side", (-1, 0, 1))
def test_interface_separation_strict_boundary_keeps_exact_small_margin(repeated, side):
    error = (
        repeated.nominal_fifth_order_contrast_error_upper_bound
        + repeated.nonlinear_initialization_contrast_error_upper_bound
    )
    error += (
        2 * repeated.repeated.nonlinear_source_error_upper_bound
        + 16 * repeated.repeated.readout_error_bound
    )
    scale = Q(7, 5) ** 3
    upper = -error / scale - side * Q(1, 2**600)
    result = owner._bound_repeated_collective_interface(
        base_cubic_lower=upper - 1, base_cubic_upper=upper
    )
    assert result.interface_separation_margin == side * scale / 2**600
    assert result.interface_separation_sufficient is (side > 0)
    assert result.conditional_repeated_interface_sufficient is (side > 0)
    assert result.status == (
        "conditional_repeated_interface_sufficient" if side > 0 else "bounds_only"
    )


@pytest.mark.parametrize("key", ("base_cubic_lower", "base_cubic_upper"))
@pytest.mark.parametrize("bad", (True, float("inf"), Decimal("1e-400")))
def test_supplied_coefficient_admitted_before_fidelity(key, bad, monkeypatch):
    monkeypatch.setattr(
        owner, "_bound_collective_interface", lambda **_: pytest.fail("late admission")
    )
    with pytest.raises((TypeError, ValueError)):
        owner._bound_repeated_collective_interface(
            **({"base_cubic_lower": -1, "base_cubic_upper": 1} | {key: bad})
        )


def test_unordered_coefficient_rejected_without_fidelity(monkeypatch):
    monkeypatch.setattr(
        owner, "_bound_collective_interface", lambda **_: pytest.fail("late admission")
    )
    with pytest.raises(ValueError, match="ordered"):
        owner._bound_repeated_collective_interface(
            base_cubic_lower=1, base_cubic_upper=0
        )
