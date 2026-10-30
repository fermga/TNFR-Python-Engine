"""Independent exact controls for the connected versus matched-cut law.

Only free-symbol complete-law identities and manufactured algebraic witnesses
are consumed. No TNFR source reader, stored preparation, field evaluator,
integrator, event or retained-response record is imported or executed.
Periodic motion below is an analytical consequence of a regular compact
energy oval, not a numerically integrated or retained-source trajectory.
"""

from types import SimpleNamespace

import sympy as s


def _zero(expression):
    assert s.cancel(s.trigsimp(s.expand(expression))) == 0


def _law(forms, phases, capacities, edges):
    """Independently differentiate reciprocal edge storage in both rows."""
    degrees = tuple(sum(i in edge[:2] for edge in edges) for i in range(len(forms)))
    form_storage = sum(
        weight * (forms[i] - forms[j]) ** 2 / 2 for i, j, weight in edges
    )
    phase_storage = sum(1 - s.cos(phases[j] - phases[i]) for i, j, _ in edges)
    storage = form_storage + phase_storage
    mobility = tuple(
        nu / (s.pi * degree) if degree else s.S.Zero
        for nu, degree in zip(capacities, degrees, strict=True)
    )
    form_rates = tuple(-m * s.diff(storage, q) for m, q in zip(mobility, phases))
    phase_rates = tuple(m * s.diff(storage, q) for m, q in zip(mobility, forms))
    return SimpleNamespace(
        forms=forms,
        phases=phases,
        capacities=capacities,
        edges=edges,
        degrees=degrees,
        mobility=mobility,
        form_storage=form_storage,
        storage=storage,
        coordinates=forms + phases,
        field=form_rates + phase_rates,
    )


def _derivative(expression, law):
    return sum(
        s.diff(expression, coordinate) * rate
        for coordinate, rate in zip(law.coordinates, law.field, strict=True)
    )


def _angle_rate(law, *, parent, child, weight):
    delta = law.phases[child] - law.phases[parent]
    r = law.forms[child] - law.forms[parent]
    return _derivative(s.atan2(s.sqrt(weight) * r, delta), law)


def test_same_state_scaled_bias_and_unscaled_degree_correction_from_full_storage():
    forms = s.symbols("x0:4", real=True)
    phases = s.symbols("theta0:4", real=True)
    capacities = s.symbols("nu0:4", positive=True)
    w, v, z = s.symbols("w v z", positive=True)
    host_edges = ((0, 2, v), (0, 3, z))
    connected = _law(forms, phases, capacities, ((0, 1, w),) + host_edges)
    cut = _law(forms, phases, capacities, host_edges)
    r, delta = forms[1] - forms[0], phases[1] - phases[0]
    current = sum(s.sin(phases[j] - phases[0]) for _, j, _ in host_edges)
    gradient = sum(weight * (forms[0] - forms[j]) for _, j, weight in host_edges)
    radius_squared = delta**2 + w * r**2
    connected_rate = _angle_rate(connected, parent=0, child=1, weight=w)
    cut_rate = _angle_rate(cut, parent=0, child=1, weight=w)
    parent, child = connected.mobility[:2]
    cut_parent = cut.mobility[0]
    degree_ratio = s.Rational(connected.degrees[0] - 1, connected.degrees[0])
    _zero(parent - degree_ratio * cut_parent)
    internal = -(parent + child) * s.sqrt(w) * (delta * s.sin(delta) + w * r**2)
    _zero(radius_squared * (connected_rate - degree_ratio * cut_rate) - internal)
    renormalization = (
        (cut_parent - parent) * s.sqrt(w) * (delta * current - r * gradient)
    )
    _zero(radius_squared * (connected_rate - cut_rate) - internal - renormalization)
    # This is a same-state vector-field identity. Scaling the cut rate is
    # algebra, not permission to change its clock or compare unequal times.


def test_complete_three_node_cut_can_be_strictly_faster_at_the_same_state():
    forms = s.symbols("x0:3", real=True)
    phases = s.symbols("theta0:3", real=True)
    capacities = (s.Integer(2), s.Integer(1), s.Integer(1))
    connected = _law(forms, phases, capacities, ((0, 1, s.S.One), (0, 2, s.S.One)))
    cut = _law(forms, phases, capacities, ((0, 2, s.S.One),))
    excess = s.symbols("excess", positive=True)
    remote_form = 2 + excess
    preparation = dict(zip(forms + phases, (0, 1, remote_form, 0, 0, 0), strict=True))
    connected_rate = _angle_rate(connected, parent=0, child=1, weight=s.S.One).subs(
        preparation
    )
    cut_rate = _angle_rate(cut, parent=0, child=1, weight=s.S.One).subs(preparation)
    _zero(connected_rate + (remote_form + 2) / s.pi)
    _zero(cut_rate + 2 * remote_form / s.pi)
    _zero(connected_rate - cut_rate - excess / s.pi)
    assert (excess / s.pi).is_positive
    # Both rates are negative; cut is more clockwise. Smoothness and R=1
    # imply this ordering persists on some short interval, not a specified
    # half-unit window. All primitive phase differences start at zero.


def _nonuniform_host():
    forms = s.symbols("x0:4", real=True)
    phases = s.symbols("theta0:4", real=True)
    A, B, C, child_capacity = s.symbols("A B C child_capacity", positive=True)
    v, z = s.symbols("v z", positive=True)
    # The host path has degrees (1,2,1), so these capacities give the
    # genuine cut weights d_i/nu_i=(A,B,C), not a supplied uniform mean.
    law = _law(
        forms, phases, (1 / A, 2 / B, 1 / C, child_capacity), ((0, 1, v), (1, 2, z))
    )
    return law, (A, B, C), (v, z)


def test_host_conserved_quantities_use_cut_weights_and_exclude_the_frozen_child():
    law, rho, _ = _nonuniform_host()
    actual_weights = tuple(
        s.cancel(d / nu) for d, nu in zip(law.degrees, law.capacities)
    )
    assert actual_weights == rho + (s.S.Zero,)
    assert law.field[3] == law.field[7] == 0
    for coordinates in (law.forms, law.phases):
        mean = sum(weight * q for weight, q in zip(rho, coordinates[:3])) / sum(rho)
        _zero(_derivative(mean, law))
    _zero(_derivative(law.storage, law))
    # The isolate's original coordinates remain available separately; zero
    # host-mean weight neither resets them nor supplies intrinsic storage.
    assert law.storage.diff(law.forms[3]) == law.storage.diff(law.phases[3]) == 0


def test_grounded_form_observation_bound_has_an_exact_square_remainder_and_is_sharp():
    law, rho, (v, z) = _nonuniform_host()
    A, B, C = rho
    total = sum(rho)
    laplacian = s.hessian(law.form_storage, law.forms[:3])
    ell = s.Matrix((1 - A / total, -B / total, -C / total))
    grounded = laplacian[:2, :2].inv() * ell[:2, :]
    potential = s.Matrix((grounded[0], grounded[1], 0))
    for entry in laplacian * potential - ell:
        _zero(entry)
    resistance = (ell.T * potential)[0]
    expected = ((B + C) ** 2 / v + C**2 / z) / total**2
    _zero(resistance - expected)
    assert expected.is_positive
    observation = (ell.T * s.Matrix(law.forms[:3]))[0]
    first, second = law.forms[0] - law.forms[1], law.forms[1] - law.forms[2]
    remainder = (z * (B + C) * second - v * C * first) ** 2 / (total**2 * v * z)
    _zero(2 * law.form_storage * resistance - observation**2 - remainder)
    assert remainder.is_nonnegative
    # The complete H adds nonnegative bare phase storage. Equality is
    # attained when forms follow the grounded potential and phases agree.
    scale, common_form, common_phase = s.symbols(
        "scale common_form common_phase", real=True
    )
    equality_state = {
        q: scale * potential[i] + common_form for i, q in enumerate(law.forms[:3])
    }
    equality_state.update({q: common_phase for q in law.phases[:3]})
    _zero((2 * law.storage * resistance - observation**2).subs(equality_state))


def test_form_separator_equality_can_have_zero_original_marked_radius():
    law, rho, (v, z) = _nonuniform_host()
    A, B, C = rho
    total = sum(rho)
    ell = s.Matrix((1 - A / total, -B / total, -C / total))
    laplacian = s.hessian(law.form_storage, law.forms[:3])
    grounded = laplacian[:2, :2].inv() * ell[:2, :]
    potential = s.Matrix((grounded[0], grounded[1], 0))
    resistance = (ell.T * potential)[0]
    positive_resistance = ((B + C) ** 2 / v + C**2 / z) / total**2
    _zero(resistance - positive_resistance)
    assert positive_resistance.is_positive
    common_form, common_phase = s.symbols("common_form common_phase", real=True)
    state = {q: potential[i] + common_form for i, q in enumerate(law.forms[:3])}
    state[law.forms[3]] = potential[0] + common_form
    state.update({q: common_phase for q in law.phases})
    host_mean = sum(weight * q for weight, q in zip(rho, law.forms[:3])) / total
    offset = (law.forms[3] - host_mean).subs(state, simultaneous=True)
    host_energy = law.storage.subs(state, simultaneous=True)
    _zero(offset - resistance)
    _zero(2 * host_energy - resistance)
    _zero(offset**2 - 2 * host_energy * resistance)
    for coordinates in (law.forms, law.phases):
        _zero((coordinates[3] - coordinates[0]).subs(state, simultaneous=True))
    # This positive-energy host saturates the form gate, yet its child is
    # exactly at the parent in both marked coordinates. Replacing the
    # strict gate by >= would admit an undefined original angle at entry.


def test_strict_form_halfplane_gives_nonzero_radius_and_a_single_smooth_angle_chart():
    width, gap, weight = s.symbols("width gap weight", positive=True)
    fraction = s.symbols("fraction", nonnegative=True)
    delta = s.symbols("delta", real=True)
    # When child-minus-host-mean=width+gap and parent-minus-host-mean is
    # width*(1-2*fraction), 0<=fraction<=1 describes the admitted interval.
    r = width + gap - width * (1 - 2 * fraction)
    residual = delta**2 + 4 * weight * width * fraction * (gap + width * fraction)
    _zero(delta**2 + weight * r**2 - weight * gap**2 - residual)
    assert residual.is_nonnegative
    vertical = s.symbols("vertical", positive=True)
    horizontal_rate, vertical_rate = s.symbols(
        "horizontal_rate vertical_rate", real=True
    )
    angle = s.pi / 2 - s.atan(delta / vertical)
    derivative = (
        s.diff(angle, delta) * horizontal_rate + s.diff(angle, vertical) * vertical_rate
    )
    _zero(
        derivative * (delta**2 + vertical**2)
        - delta * vertical_rate
        + vertical * horizontal_rate
    )
    # This continuous chart lies strictly in (0,pi); any net increment has
    # absolute value <pi. Its total variation need not be <pi. The lower
    # half-plane follows by reflecting the vertical coordinate.


def test_complete_cut_can_have_a_periodic_clockwise_marked_loop():
    forms = s.symbols("x0:3", real=True)
    phases = s.symbols("theta0:3", real=True)
    nu, child_capacity = s.symbols("nu child_capacity", positive=True)
    law = _law(forms, phases, (nu, nu, child_capacity), ((0, 1, s.S.One),))
    r, delta, common_form, common_phase = s.symbols(
        "r delta common_form common_phase", real=True
    )
    state = {
        forms[0]: common_form - r,
        forms[1]: common_form + r,
        forms[2]: common_form,
        phases[0]: common_phase - delta,
        phases[1]: common_phase + delta,
        phases[2]: common_phase,
    }
    r_dot = _derivative(forms[2] - forms[0], law).subs(state, simultaneous=True)
    delta_dot = _derivative(phases[2] - phases[0], law).subs(state, simultaneous=True)
    mobility = nu / s.pi
    _zero(r_dot + mobility * s.sin(2 * delta))
    _zero(delta_dot - 2 * mobility * r)
    energy = law.storage.subs(state, simultaneous=True)
    _zero(energy - 2 * r**2 - 1 + s.cos(2 * delta))
    _zero(energy - 2 * (r**2 + s.sin(delta) ** 2))
    _zero(s.diff(energy, r) * r_dot + s.diff(energy, delta) * delta_dot)
    numerator = delta * r_dot - r * delta_dot
    _zero(numerator + mobility * (delta * s.sin(2 * delta) + 2 * r**2))
    for coordinates in (forms, phases):
        center = coordinates[2] - (coordinates[0] + coordinates[1]) / 2
        _zero(center.subs(state, simultaneous=True))
    # On the central 0<H<2 oval, |delta|<pi/2 and the only possible
    # stationary point r=delta=0 is excluded. Hence the smooth compact
    # regular oval is periodic and the numerator is strictly negative.
    # These are conditional analytical facts, not a period or horizon fit.
    # Here the fixed ellipse center is zero, so its strict origin-exclusion
    # gate Q>2H fails for every positive energy, consistently with the loop.
    assert energy.subs({r: 0, delta: 0}) == 0
    assert energy.subs({r: 0, delta: s.pi / 2}) == 2


def test_joint_ellipse_uses_separate_form_and_phase_storage_without_double_counting():
    law, rho, (v, z) = _nonuniform_host()
    _, B, C = rho
    total = sum(rho)
    form_gaps = (law.forms[0] - law.forms[1], law.forms[1] - law.forms[2])
    phase_gaps = (law.phases[0] - law.phases[1], law.phases[1] - law.phases[2])
    form_observation = ((B + C) * form_gaps[0] + C * form_gaps[1]) / total
    phase_observation = ((B + C) * phase_gaps[0] + C * phase_gaps[1]) / total
    form_resistance = ((B + C) ** 2 / v + C**2 / z) / total**2
    phase_resistance = ((B + C) ** 2 + C**2) / total**2
    form_remainder = (z * (B + C) * form_gaps[1] - v * C * form_gaps[0]) ** 2 / (
        total**2 * v * z
    )
    phase_remainder = ((B + C) * phase_gaps[1] - C * phase_gaps[0]) ** 2 / total**2
    allocated_quadratic = (
        2 * law.form_storage + 4 * sum(q**2 for q in phase_gaps) / s.pi**2
    )
    ellipse = form_observation**2 / form_resistance + 4 * phase_observation**2 / (
        s.pi**2 * phase_resistance
    )
    observation_weight = s.symbols("observation_weight", positive=True)
    marked_offset = s.Matrix(
        (phase_observation, s.sqrt(observation_weight) * form_observation)
    )
    shape = s.diag(s.pi**2 * phase_resistance / 4, observation_weight * form_resistance)
    _zero((marked_offset.T * shape.inv() * marked_offset)[0] - ellipse)
    residual = form_remainder / form_resistance + 4 * phase_remainder / (
        s.pi**2 * phase_resistance
    )
    _zero(allocated_quadratic - ellipse - residual)
    assert residual.is_nonnegative
    # Only on each actual lifted edge chart |q|<=pi does
    # 1-cos(q)>=2*q**2/pi**2 compare allocated_quadratic with 2*H.
    # It is not a global estimate on arbitrary real phase lifts.


def test_ellipse_separation_normal_has_exact_cauchy_remainders():
    first_axis, second_axis = s.symbols("first_axis second_axis", positive=True)
    z1, z2, e1, e2 = s.symbols("z1 z2 e1 e2", real=True)
    center_norm = z1**2 / first_axis + z2**2 / second_axis
    error_norm = e1**2 / first_axis + e2**2 / second_axis
    signed_error = z1 * e1 / first_axis + z2 * e2 / second_axis
    remainder = (z1 * e2 - z2 * e1) ** 2 / (first_axis * second_axis)
    _zero(center_norm * error_norm - signed_error**2 - remainder)
    assert remainder.is_nonnegative
    normal = (z1 / first_axis, z2 / second_axis)
    observed = (z1 + e1, z2 + e2)
    projection = sum(a * b for a, b in zip(normal, observed))
    _zero(projection - center_norm - signed_error)
    radius_squared = sum(value**2 for value in observed)
    normal_squared = sum(value**2 for value in normal)
    radius_remainder = (normal[0] * observed[1] - normal[1] * observed[0]) ** 2
    _zero(normal_squared * radius_squared - projection**2 - radius_remainder)
    assert radius_remainder.is_nonnegative
    # With error_norm<=2H and center_norm>2H, the normal D^-1*z gives
    # projection>=center_norm-sqrt(2H*center_norm)>0. Its fixed open
    # half-plane yields nonzero original radius and |net angle|<pi;
    # neither the principal angle nor total variation replaces that lift.
