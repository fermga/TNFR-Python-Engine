"""Static controls for a critical six-child parent and one marked C9 neighbor.

Independent edge sums and projection algebra retain both complete rows, the
relative means and hidden port state. No source, root or trajectory is evaluated.
"""

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.physics._sine_class_constitutive_comparison import _edge_law


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_organized_response_evaluation():
        yield


@pytest.fixture(scope="module")
def geometry():
    eta, epsilon, gamma = sp.symbols("eta epsilon gamma", positive=True)
    old_edges = (
        [
            (9 * child + slot, 9 * child + (slot + 1) % 9, 1)
            for child in range(7)
            for slot in range(9)
        ]
        + [
            (9 * child + slot, 9 * ((child + 1) % 5) + slot, 1)
            for child in range(5)
            for slot in range(9)
        ]
        + [
            (9 * child + slot, 45 + slot, eta)
            for child in range(5)
            for slot in range(9)
        ]
    )
    old = sp.zeros(63)
    for left, right, weight in old_edges:
        old[left, left] += weight
        old[right, right] += weight
        old[left, right] -= weight
        old[right, left] -= weight
    boundary = sp.zeros(9, 63)
    for slot in range(9):
        boundary[slot, 18 + slot], boundary[slot, 54 + slot] = -1, 1
    change = boundary.T * boundary
    lift = sp.kronecker_product(sp.eye(7), sp.ones(9, 1))
    wheel = sp.zeros(7)
    for child in range(5):
        wheel[child, child] = 2 + eta
        wheel[child, (child - 1) % 5] = -1
        wheel[child, (child + 1) % 5] = -1
        wheel[child, 5] = wheel[5, child] = -eta
    wheel[5, 5] = 5 * eta
    wheel[2, 2] += epsilon
    wheel[6, 6] += epsilon
    wheel[2, 6] = wheel[6, 2] = -epsilon
    return eta, epsilon, gamma, old, change, boundary, lift, wheel, old_edges


def _critical_geometry(lift):
    shape = list(sp.symbols("shape0:4", real=True))
    shape.append(-sum(shape))
    wheel_mode = sp.Matrix(
        [sp.Rational(1, 6) + value for value in shape] + [-sp.Rational(5, 6), 0]
    )
    mode = lift * wheel_mode
    relative = lift * sp.Matrix([-sp.Rational(1, 7)] * 6 + [sp.Rational(6, 7)])
    mean_difference = sp.Matrix([[-sp.Rational(1, 54)] * 54 + [sp.Rational(1, 9)] * 9])
    raw_origin = sp.Matrix(
        [[sp.Rational(1, 45)] * 45 + [-sp.Rational(1, 9)] * 9 + [0] * 9]
    )
    norm = sp.expand((mode.T * mode)[0])
    return shape, wheel_mode, mode, relative, mean_difference, raw_origin, norm


def test_full_63_edge_carrier_and_unchanged_clock_balance(geometry):
    """Slot-constant current and form rows close on the actual seven-child graph."""
    eta, epsilon, gamma, old, change, _, lift, wheel, edges = geometry
    laplacian = old + epsilon * change
    assert len(edges) + 9 == 162
    assert laplacian * sp.ones(63, 1) == sp.zeros(63, 1)
    assert laplacian * lift == lift * wheel
    forms = sp.Matrix(sp.symbols("form0:7", real=True))
    rim = sp.symbols("rim0:5", real=True)
    spokes = sp.symbols("spoke0:5", real=True)
    internal, port = sp.symbols("internal port", real=True)
    current = sp.zeros(63, 1)
    # Every internal oriented C9 edge has the same actual carrier current;
    # odd periodicity includes the closing edge, so the two contributions cancel.
    for index, (left, right, weight) in enumerate(edges):
        value = (
            internal
            if index < 63
            else rim[(index - 63) // 9] if index < 108 else spokes[(index - 108) // 9]
        )
        current[left] += weight * value
        current[right] -= weight * value
    for slot in range(9):
        current[18 + slot] += epsilon * port
        current[54 + slot] -= epsilon * port
    coarse_current = sp.Matrix(
        [
            rim[c]
            - rim[(c - 1) % 5]
            + eta * spokes[c]
            + (epsilon * port if c == 2 else 0)
            for c in range(5)
        ]
        + [-eta * sum(spokes), -epsilon * port]
    )
    assert sp.simplify(current - lift * coarse_current) == sp.zeros(63, 1)
    full_form_rate = (-laplacian * lift * forms + gamma * current) / 4
    full_phase_rate = gamma * laplacian * lift * forms / 4
    assert sp.simplify(
        full_form_rate - lift * (-wheel * forms + gamma * coarse_current) / 4
    ) == sp.zeros(63, 1)
    assert sp.simplify(
        full_phase_rate - lift * (gamma * wheel * forms) / 4
    ) == sp.zeros(63, 1)
    gradient = laplacian * lift * forms
    storage_rate = (gradient.T * full_form_rate - current.T * full_phase_rate)[0]
    assert sp.expand(storage_rate + (gradient.T * gradient)[0] / 4) == 0
    assert (lift.T * lift) == 9 * sp.eye(7)

    # Updating only the form row leaves an uncancelled signed work term.
    wrong_phase_rate = gamma * old * lift * forms / 4
    defect = sp.expand((current.T * (full_phase_rate - wrong_phase_rate))[0])
    zero_internal = dict.fromkeys((*rim, *spokes), 0)
    unit_port_form = dict.fromkeys(forms, 0) | {forms[2]: 1}
    assert (
        sp.simplify(defect.subs(zero_internal | unit_port_form))
        == 9 * gamma * epsilon**2 * port / 2
    )


def test_exact_parent_neighbor_means_raw_origin_and_event_work(geometry):
    """Six-to-one mean weights differ from equal-sized pair factors."""
    _, epsilon, gamma, old, change, _, lift, wheel, _ = geometry
    forms = sp.Matrix(sp.symbols("form0:7", real=True))
    port = sp.symbols("port", real=True)
    external_current = sp.zeros(7, 1)
    external_current[2], external_current[6] = epsilon * port, -epsilon * port
    # Only boundary terms survive summation over the complete parent.
    old_wheel = (lift.T * old * lift) / 9
    form_rate = (-wheel * forms + gamma * external_current) / 4
    phase_rate = gamma * wheel * forms / 4
    parent_form = sum(forms[:6]) / 6
    relative_form = forms[6] - parent_form
    port_deviation = forms[2] - parent_form
    form_gap = relative_form - port_deviation
    parent_rate = sum(form_rate[:6]) / 6
    assert sp.simplify(parent_rate - epsilon * (form_gap + gamma * port) / 24) == 0
    relative_rate = form_rate[6] - parent_rate
    assert (
        sp.simplify(relative_rate + 7 * epsilon * (form_gap + gamma * port) / 24) == 0
    )
    phase_difference_rate = phase_rate[6] - sum(phase_rate[:6]) / 6
    assert sp.simplify(phase_difference_rate - 7 * gamma * epsilon * form_gap / 24) == 0
    external_phase = gamma * (wheel - old_wheel) * forms / 4
    raw_origin_rate = sum(external_phase[:5]) / 5 - external_phase[5]
    assert sp.simplify(raw_origin_rate + gamma * epsilon * form_gap / 20) == 0
    assert sp.simplify(6 * parent_rate + form_rate[6]) == 0

    deformation = sp.symbols("deformation", nonnegative=True)
    potential, _ = _edge_law(sp.Rational(3, 5), sp.Rational(4, 5), deformation)
    full_forms = lift * forms
    form_jump = (full_forms.T * (epsilon * change) * full_forms)[0] / 2
    assert sp.simplify(form_jump - 9 * epsilon * (forms[6] - forms[2]) ** 2 / 2) == 0
    complete_jump = form_jump + 9 * epsilon * potential
    assert (
        sp.simplify(
            complete_jump.subs(dict.fromkeys(forms, 0)) - 9 * epsilon * potential
        )
        == 0
    )
    assert potential.subs(deformation, 0) == sp.Rational(2, 5)
    assert sp.diff(potential, deformation) > 0
    assert _edge_law(sp.Integer(1), sp.Integer(0), deformation)[0] == 0


def test_critical_and_mean_projections_derive_the_joint_linear_contact(geometry):
    """The center projection includes form; the raw registration row is different."""
    _, epsilon, gamma, _, change, _, lift, _, _ = geometry
    _, _, mode, relative, mean, raw, norm = _critical_geometry(lift)
    port_slope = mode[18]
    mobility = 9 * gamma**2 / (4 * norm)
    phase_columns = mode.row_join(sp.zeros(63, 1)).row_join(relative)
    form_columns = sp.zeros(63, 1).row_join(relative).row_join(sp.zeros(63, 1))
    right = phase_columns.col_join(form_columns)
    left = (mode.T / norm).row_join(gamma * mode.T / norm)
    left = left.col_join(sp.zeros(1, 63).row_join(mean))
    left = left.col_join(mean.row_join(sp.zeros(1, 63)))
    assert sp.simplify(left * right) == sp.eye(3)
    assert sp.simplify(raw * mode) == sp.ones(1, 1)
    assert (mean * relative)[0] == 1
    assert sp.simplify(sum(mode)) == 0
    # Differentiate the newly added field at its aligned zero gap: j'(0)=1.
    deformation = sp.symbols("deformation", nonnegative=True)
    _, derivatives = _edge_law(sp.Integer(1), sp.Integer(0), deformation)
    assert derivatives[0] == derivatives[2] == 0
    assert derivatives[1] == 1
    phase_change = gamma * epsilon * change * form_columns / 4
    form_change = (
        -epsilon * (change * form_columns + gamma * change * phase_columns) / 4
    )
    projected = sp.simplify(left * phase_change.col_join(form_change))
    rate = sp.Rational(7, 24)
    expected = epsilon * sp.Matrix(
        [
            [-mobility * port_slope**2, 0, mobility * port_slope],
            [rate * gamma * port_slope, -rate, -rate * gamma],
            [0, rate * gamma, 0],
        ]
    )
    assert sp.simplify(projected - expected) == sp.zeros(3)
    assert raw * phase_change == sp.Matrix([[0, -gamma * epsilon / 20, 0]])
    assert projected[0, 0] != 0
    # Freezing either separate mean deletes nonzero rows of this exact projection.
    assert projected[1, 2] == -7 * gamma * epsilon / 24
    assert projected[2, 1] == 7 * gamma * epsilon / 24


def test_positive_contact_kernel_loading_and_initialization(geometry):
    """A positive port constraint leaves one soft mode and changes its metric."""
    _, _, gamma, old, _, boundary, lift, wheel, edges = geometry
    shape, wheel_mode, mode, relative, mean, raw, norm = _critical_geometry(lift)
    port_slope = mode[18]
    # Four independent host-shape constraints model the already proved positive
    # normal block, without manufacturing an unknown critical equilibrium.
    origin_row = sp.Matrix([[sp.Rational(1, 5)] * 5 + [-1, 0]])
    normal = sp.zeros(4, 7)
    shape_basis = sp.zeros(7, 4)
    for index in range(4):
        normal[index, :] = (
            sp.Matrix(
                [[int(c == index) - sp.Rational(1, 5) for c in range(5)] + [0, 0]]
            )
            - shape[index] * origin_row
        )
        shape_basis[index, index], shape_basis[4, index] = 1, -1
    wheel_relative = sp.Matrix([-sp.Rational(1, 7)] * 6 + [sp.Rational(6, 7)])
    kernel = sp.ones(7, 1).row_join(wheel_mode).row_join(wheel_relative)
    assert sp.simplify(normal * kernel) == sp.zeros(4, 3)
    assert normal * shape_basis == sp.eye(4)
    assert sp.simplify(boundary * lift * kernel) == sp.ones(9, 1) * sp.Matrix(
        [[0, -port_slope, 1]]
    )
    # Seven internal C9 complements contribute 56 phase directions. The
    # boundary has rank one on the three old phase null directions.
    internal = sp.zeros(9)
    for left, right, _ in edges[:9]:
        internal[left, left] += 1
        internal[right, right] += 1
        internal[left, right] -= 1
        internal[right, left] -= 1
    old_wheel = lift.T * old * lift / 9
    fine_rank = 7 * internal.rank()
    normal_rank = fine_rank + normal.rank()
    assert normal_rank == 60
    added_rank = sp.Matrix([[0, -port_slope, 1]]).rank()
    assert added_rank == 1
    old_form_rank, joined_form_rank = (
        fine_rank + old_wheel.rank(),
        fine_rank + wheel.rank(),
    )
    old_centers = (63 - normal_rank) + (63 - old_form_rank) - 2
    joined_centers = (63 - normal_rank - added_rank) + (63 - joined_form_rank) - 2
    assert old_centers == 3
    assert joined_centers == 1
    assert 2 * (63 - 1) - old_centers == 121
    assert 2 * (63 - 1) - joined_centers == 123
    joined = mode + port_slope * relative
    assert sp.simplify(boundary * joined) == sp.zeros(9, 1)
    assert sp.simplify(sum(joined)) == 0
    assert sp.simplify(raw * joined) == sp.ones(1, 1)
    joined_norm = sp.expand((joined.T * joined)[0])
    assert sp.simplify(joined_norm - norm - sp.Rational(54, 7) * port_slope**2) == 0
    ratio = sp.factor(norm / joined_norm)
    assert ratio.subs(dict.fromkeys(shape[:4], 0)) == sp.Rational(35, 36)
    phase = sp.Matrix(sp.symbols("phase0:63", real=True))
    form = sp.Matrix(sp.symbols("form0:63", real=True))
    original_projection = (mode.T * (phase + gamma * form))[0] / norm
    relative_projection = (mean * (phase + gamma * form))[0]
    joined_projection = (joined.T * (phase + gamma * form))[0] / joined_norm
    assert (
        sp.simplify(
            joined_projection
            - (
                norm * original_projection
                + sp.Rational(54, 7) * port_slope * relative_projection
            )
            / joined_norm
        )
        == 0
    )


def test_marked_port_sees_critical_amplitude_and_hidden_form(geometry):
    """Matched means neither hide the soft port response nor close raw dynamics."""
    _, epsilon, gamma, _, _, _, lift, wheel, _ = geometry
    _, _, mode, _, _, raw, _ = _critical_geometry(lift)
    eta_ceiling = sp.Rational(1, 10**12)
    assert sp.Rational(1, 6) - 6 * eta_ceiling > sp.Rational(1, 7)
    displacement, port_slope = sp.symbols("displacement port_slope", real=True)
    # At zero forms and equal parent/neighbor mean phase, an odd stationary
    # port displacement s(t) has s'(0)=w2>0 and j'(0)=1 for the actual law.
    gap = -port_slope * displacement
    deformation = sp.symbols("deformation", nonnegative=True)
    current = (
        sp.sin(gap) + deformation * (sp.sin(9 * gap) - sp.sin(18 * gap) / 2) / 2187
    )
    assert sp.simplify(current.subs(displacement, -displacement) + current) == 0
    reflection = sp.zeros(7)
    for child in range(7):
        reflection[child, 4 - child if child < 5 else child] = 1
    assert reflection.T * wheel * reflection == wheel
    parent_exchange = epsilon * gamma * current / 24
    neighbor_exchange = -epsilon * gamma * current / 4
    assert (
        sp.diff(parent_exchange, displacement).subs(displacement, 0)
        == -epsilon * gamma * port_slope / 24
    )
    assert (
        sp.diff(neighbor_exchange, displacement).subs(displacement, 0)
        == epsilon * gamma * port_slope / 4
    )

    # The same phases and all the same raw (t,r,d) can instead conceal a
    # mean-zero internal form contrast. Its port value changes both mean rows.
    perturbation = sp.zeros(7, 1)
    perturbation[2], perturbation[0] = displacement, -displacement
    assert sum(perturbation[:6]) == perturbation[6] == 0
    phase_rate = gamma * wheel * perturbation / 4
    relative_phase_rate = phase_rate[6] - sum(phase_rate[:6]) / 6
    assert sp.simplify(relative_phase_rate) == -7 * epsilon * gamma * displacement / 24
    form_rate = -wheel * perturbation / 4
    relative_form_rate = form_rate[6] - sum(form_rate[:6]) / 6
    assert sp.simplify(relative_form_rate) == 7 * epsilon * displacement / 24
    assert sp.simplify((raw * lift * phase_rate)[0]) != 0
    # Its critical initialization also retains the weighted internal form;
    # observing zero raw phase is not a certificate of zero slow initialization.
    assert sp.simplify((mode.T * lift * perturbation)[0]) == 9 * displacement * (
        mode[18] - mode[0]
    )
