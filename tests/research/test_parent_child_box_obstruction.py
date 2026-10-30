"""Independent symbolic controls for the frozen Cartesian-box obstruction.

These free-symbol calculations concern necessary enclosure-image radii and
the registered absolute-sum work recipe. They evaluate no retained source,
candidate box, Picard iteration, trajectory, event or numerical parameter.
"""

import sympy as s


def _zero(expression):
    assert s.cancel(s.expand(expression)) == 0


def test_signed_full_rows_cancel_before_the_absolute_coefficient_sum():
    # Two external neighbors also share an edge. Their complete rows retain
    # its contribution, whose cancellation cannot be obtained after abs().
    ba, bc, bj, bl, clock = s.symbols("ba bc bj bl clock", positive=True)
    w, vj, vl, cross = s.symbols("w vj vl cross", positive=True)
    incidence = s.Matrix(((1, 1, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 1), (0, 0, -1, -1)))
    laplacian = incidence * s.diag(w, vj, vl, cross) * incidence.T
    mobility = s.diag(ba, bc, bj, bl) / clock
    gamma, mur = ba / (ba + bc), ba * bc / (ba + bc)
    strength = vj + vl
    port = s.Matrix((strength, 0, -vj, -vl))
    center = s.Matrix((1 - gamma, gamma, 0, 0))
    coefficients = []
    for node, beta, weight in ((2, bj, vj), (3, bl, vl)):
        contrast = s.eye(4)[:, node] - center
        full_row = contrast.T * mobility * laplacian
        exact_numerator = beta * laplacian[node, :] - mur * port.T
        for index in range(4):
            _zero(clock * full_row[index] - exact_numerator[index])
        _zero(full_row[1])
        _zero(full_row[0] + (beta * weight + mur * strength) / clock)
        coefficients.append(full_row)
    parent_absolute_sum = (bj * vj + bl * vl + 2 * mur * strength) / clock
    _zero(-sum(row[0] for row in coefficients) - parent_absolute_sum)
    assert parent_absolute_sum.is_positive
    # A second, non-child coefficient can vanish through its signed rows.
    cancellation = {cross: mur * vl / bj}
    _zero(coefficients[0][3].subs(cancellation))
    absolute_before_combining = (bj * cross + mur * vl) / clock
    _zero(absolute_before_combining.subs(cancellation) - 2 * mur * vl / clock)
    assert (2 * mur * vl / clock).is_positive


def test_picard_radius_slacks_force_the_registered_joint_work_square_remainder():
    duration, gamma, m, kappa = s.symbols("duration gamma m kappa", positive=True)
    # The initial point's straight Picard image requires each symmetric
    # outer radius >= T*abs(xdot_i0), even if the containing tube is asymmetric.
    # These v_i are absolute initial rates, not actual displacements.
    speeds = s.symbols("speed0:3", nonnegative=True)
    slacks = s.symbols("slack0:3", nonnegative=True)
    rows = (
        s.symbols("h00 h01 h02", nonnegative=True),
        s.symbols("h10 h11 h12", nonnegative=True),
    )
    radii = tuple(duration * v + slack for v, slack in zip(speeds, slacks, strict=True))
    phase_extra, delta_v = s.symbols("phase_extra delta_v", nonnegative=True)
    Kx = sum(h * v for row in rows for h, v in zip(row, speeds, strict=True))
    delta_z = phase_extra + sum(
        h * radius for row in rows for h, radius in zip(row, radii, strict=True)
    )
    remainder = phase_extra + sum(
        h * slack for row in rows for h, slack in zip(row, slacks, strict=True)
    )
    _zero(delta_z - duration * Kx - remainder)
    assert remainder.is_nonnegative
    chosen_rate_increment_squared = gamma**2 * (delta_v**2 / kappa + delta_z**2 / m)
    forced_rate_increment_squared = gamma**2 * duration**2 * Kx**2 / m
    square_remainder = gamma**2 * (
        delta_v**2 / kappa + remainder * (2 * duration * Kx + remainder) / m
    )
    _zero(
        chosen_rate_increment_squared - forced_rate_increment_squared - square_remainder
    )
    assert square_remainder.is_nonnegative
    # Nonnegative square roots and multiplication by T give the necessary
    # chosen-budget floor G0*T + gamma*Kx*T**2/sqrt(m), never a work lower
    # bound on the actual solution.


def test_minimum_duration_rejects_all_larger_horizons_for_each_nonnegative_floor():
    initial_rate, coefficient, increment = s.symbols(
        "initial_rate coefficient increment", nonnegative=True
    )
    minimum = s.symbols("minimum", positive=True)
    time = s.symbols("time", nonnegative=True)
    floor = initial_rate * time + coefficient * time**2
    difference = floor.subs(time, minimum + increment) - floor.subs(time, minimum)
    remainder = increment * (initial_rate + coefficient * (2 * minimum + increment))
    _zero(difference - remainder)
    assert remainder.is_nonnegative
    # Both Kx and the activated row-saturation gate have this form. Comparing
    # the floor at the necessary Tmin with J_signal covers every T>=Tmin;
    # it neither adopts Tmin as a horizon nor proves a feasible box.


def test_unsaturated_pair_box_cannot_close_once_the_scaled_duration_reaches_one():
    duration, k, form_slack, phase_slack = s.symbols(
        "duration k form_slack phase_slack", positive=True
    )
    ratio, excess = s.symbols("ratio excess", nonnegative=True)
    phase_radius = 1 / (1 + ratio)  # 0 < b_pair <= 1.
    weight = (1 + excess) / (duration**2 * k**2)
    # Under this branch min(1,b_pair)=b_pair. First impose the strict form
    # bound, then show that the required phase radius already exceeds b_pair.
    form_radius = duration * k * phase_radius + form_slack
    required_phase_radius = duration * k * weight * form_radius + phase_slack
    contradiction = (
        excess * phase_radius + duration * k * weight * form_slack + phase_slack
    )
    _zero(required_phase_radius - phase_radius - contradiction)
    assert contradiction.is_positive
    _zero(duration**2 * k**2 * weight - 1 - excess)


def test_saturated_pair_lower_bounds_force_parent_phase_radius_beyond_one():
    duration, ma, mc = s.symbols("duration ma mc", positive=True)
    form_a_slack, form_c_slack, phase_slack = s.symbols(
        "form_a_slack form_c_slack phase_slack", positive=True
    )
    activation_excess = s.symbols("activation_excess", nonnegative=True)
    k, gamma = ma + mc, ma / (ma + mc)
    # The already-proved b_pair>1 branch gives a_a>T*ma and a_c>T*mc.
    form_a = duration * ma + form_a_slack
    form_c = duration * mc + form_c_slack
    weight = (1 + activation_excess) / (gamma * duration**2 * k**2)
    parent_phase = duration * ma * weight * (form_a + form_c) + phase_slack
    positive_remainder = (
        activation_excess
        + duration * ma * weight * (form_a_slack + form_c_slack)
        + phase_slack
    )
    _zero(parent_phase - 1 - positive_remainder)
    assert positive_remainder.is_positive
    _zero(gamma * duration**2 * k**2 * weight - 1 - activation_excess)


def test_parent_degree_saturation_forces_the_separate_work_coefficient():
    duration, ma, degree, Ha = s.symbols("duration ma degree Ha", positive=True)
    excess_sum, radius_slack, other_columns, phase_extra = s.symbols(
        "excess_sum radius_slack other_columns phase_extra", positive=True
    )
    initial_sine_sum = s.symbols("initial_sine_sum", nonnegative=True)
    # b_a>1 and each b_j>0 make every min(2,b_a+b_j)>1. Their sum is
    # d_a+excess_sum, so the registered outer minimum reaches d_a exactly.
    sine_bound = s.Min(degree, initial_sine_sum + degree + excess_sum)
    assert sine_bound == degree
    parent_radius = duration * ma * sine_bound + radius_slack
    delta_z = Ha * parent_radius + other_columns + phase_extra
    remainder = Ha * radius_slack + other_columns + phase_extra
    _zero(delta_z - duration * ma * degree * Ha - remainder)
    assert remainder.is_positive
    # Ha is the already-combined signed parent-column absolute sum. The
    # resulting gate applies only after both pair-saturation premises hold.
