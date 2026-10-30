"""Exact storage controls for the frozen signed-transport analytical study.

These free-symbol graph identities evaluate no retained source, solver,
trajectory or event. The final witness compares two admissible full states;
it does not assert a returned-coefficient history on one orbit.
"""

from types import SimpleNamespace

import sympy as s


def _zero(expression):
    assert s.cancel(s.trigsimp(s.expand_trig(s.expand(expression)))) == 0


def _edge_partition(*, two_external):
    """Assemble reciprocal edge storage in the invertible pair-center chart."""
    r, delta, delta_star = s.symbols("r delta delta_star", real=True)
    gamma, w, v, z, h = s.symbols("gamma w v z h", positive=True)
    weights = (v, z) if two_external else (v,)
    y = s.symbols(f"y0:{len(weights)}", real=True)
    alpha = s.symbols(f"alpha0:{len(weights)}", real=True)
    # y_j=X-x_j and alpha_j=theta_j-Theta. Since x_a=X-gamma*r,
    # the original a-j form difference is y_j-gamma*r. The external
    # phase difference is alpha_j+gamma*delta, with no host held fixed.
    rest = (
        h * (y[0] - y[1]) ** 2 / 2 + 1 - s.cos(alpha[1] - alpha[0])
        if two_external
        else s.S.Zero
    )
    full = rest + w * r**2 / 2 + 1 - s.cos(delta)
    full += sum(
        weight * (value - gamma * r) ** 2 / 2 + 1 - s.cos(angle + gamma * delta)
        for weight, value, angle in zip(weights, y, alpha, strict=True)
    )
    sigma = sum(weights)
    V = sum(weight * value for weight, value in zip(weights, y, strict=True))
    C, S = sum(s.cos(a) for a in alpha), sum(s.sin(a) for a in alpha)
    kappa = w + gamma**2 * sigma

    def potential(angle):
        return (
            1 - s.cos(angle) + C * (1 - s.cos(gamma * angle)) + S * s.sin(gamma * angle)
        )

    hidden = rest + sum(
        weight * (value - V / sigma) ** 2 / 2
        for weight, value in zip(weights, y, strict=True)
    )
    relative = (
        kappa * (r - gamma * V / kappa) ** 2 / 2
        + potential(delta)
        - potential(delta_star)
    )
    boundary = w * V**2 / (2 * sigma * kappa) + len(weights) - C + potential(delta_star)
    return SimpleNamespace(
        r=r,
        delta=delta,
        delta_star=delta_star,
        gamma=gamma,
        w=w,
        h=h,
        weights=weights,
        y=y,
        alpha=alpha,
        sigma=sigma,
        V=V,
        C=C,
        kappa=kappa,
        rest=rest,
        full=full,
        hidden=hidden,
        relative=relative,
        boundary=boundary,
    )


def test_full_edge_storage_retains_external_variance_and_residual_edges():
    p = _edge_partition(two_external=True)
    # The four-node graph has a pendant marked child, two external parent
    # neighbors, and an edge between those neighbors. Neither its environmental
    # variance nor that edge storage is determined by V alone.
    _zero(p.full - p.relative - p.hidden - p.boundary)
    v, z = p.weights
    _zero(p.hidden - p.rest - v * z * (p.y[0] - p.y[1]) ** 2 / (2 * (v + z)))


def test_physical_boundary_and_hidden_terms_are_nonnegative_and_give_upper_bound():
    p = _edge_partition(two_external=True)
    # These exact square representations use actual phasors, not arbitrary
    # unrelated numbers C and S. Every coefficient displayed is positive.
    boundary_squares = p.w * p.V**2 / (2 * p.sigma * p.kappa)
    boundary_squares += 2 * s.sin(p.delta_star / 2) ** 2
    boundary_squares += sum(
        2 * s.sin((angle + p.gamma * p.delta_star) / 2) ** 2 for angle in p.alpha
    )
    hidden_squares = p.h * (p.y[0] - p.y[1]) ** 2 / 2
    hidden_squares += 2 * s.sin((p.alpha[1] - p.alpha[0]) / 2) ** 2
    hidden_squares += sum(
        weight * (value - p.V / p.sigma) ** 2 / 2
        for weight, value in zip(p.weights, p.y, strict=True)
    )
    _zero(p.boundary - boundary_squares)
    _zero(p.hidden - hidden_squares)
    # Thus L <= H-Psi. Conservation turns this into the one-sided endpoint
    # upper bound L(t) <= L(0)+H_hidden(0)+Psi(0)-Psi(t). It supplies no
    # positive lower bound when the hidden storage may increase.
    _zero((p.full - p.boundary) - p.relative - hidden_squares)


def test_three_node_exception_forces_relative_storage_return_with_coefficients():
    p = _edge_partition(two_external=False)
    _zero(p.hidden)
    _zero(p.full - p.relative - p.boundary)
    next_r, next_delta = s.symbols("next_r next_delta", real=True)
    next_state = {p.r: next_r, p.delta: next_delta}
    # With the same coefficients and reference branch, the boundary term
    # agrees at both endpoints. Hence conservation of the complete H forces
    # L to return; an independently prescribed coefficient loop is not a
    # counterexample for this complete three-node law.
    delta_L = p.relative.subs(next_state, simultaneous=True) - p.relative
    delta_H = p.full.subs(next_state, simultaneous=True) - p.full
    _zero(delta_L - delta_H)


def test_four_node_complete_law_has_same_observations_and_opposite_signed_work():
    forms = s.symbols("x0:4", real=True)
    phases = s.symbols("theta0:4", real=True)
    w, v, z, u, alpha = s.symbols("w v z u alpha", positive=True)
    edges = ((0, 1, w), (0, 2, v), (2, 3, z))
    full = sum(
        weight * (forms[j] - forms[i]) ** 2 / 2 + 1 - s.cos(phases[j] - phases[i])
        for i, j, weight in edges
    )
    # All mobilities are one: choose held positive nu_i=pi*d_i, rather than
    # silently replacing the model's degree-normalized capacity convention.
    degrees = tuple(sum(i in edge[:2] for edge in edges) for i in range(4))
    capacities = tuple(s.pi * degree for degree in degrees)
    rho = tuple(degree / capacity for degree, capacity in zip(degrees, capacities))
    form_rates = tuple(-s.diff(full, phase) for phase in phases)
    phase_rates = tuple(s.diff(full, form) for form in forms)

    def derivative(expression):
        return sum(
            s.diff(expression, coordinate) * rate
            for coordinate, rate in zip(forms + phases, form_rates + phase_rates)
        )

    _zero(derivative(full))
    for coordinates in (forms, phases):
        _zero(derivative(sum(a * b for a, b in zip(rho, coordinates))))
    r, delta = forms[1] - forms[0], phases[1] - phases[0]
    X, Theta = (forms[0] + forms[1]) / 2, (phases[0] + phases[1]) / 2
    V, external_phase = v * (X - forms[2]), phases[2] - Theta
    C, S = s.cos(external_phase), s.sin(external_phase)
    rest = z * (forms[3] - forms[2]) ** 2 / 2 + 1 - s.cos(phases[3] - phases[2])
    # Independently obtain K by subtracting the environmental edge storage.
    K = full - rest - V**2 / (2 * v) - (1 - C)
    kappa = w + v / 4
    expected_storage = kappa * u**2 / 2 + 1 - s.cos(alpha)

    # At V=S=0,C=1 the reference is (r*,delta*)=(0,0). All state and
    # coefficient derivatives of K at that reference vanish, so K_min_dot=0.
    rr, dd, VV, CC, SS = s.symbols("rr dd VV CC SS", real=True)
    reduced = kappa * rr**2 / 2 - VV * rr / 2 + 1 - s.cos(dd)
    reduced += CC * (1 - s.cos(dd / 2)) + SS * s.sin(dd / 2)
    minimum = {rr: 0, dd: 0, VV: 0, CC: 1, SS: 0}
    for coordinate in (rr, dd, VV, CC, SS):
        _zero(s.diff(reduced, coordinate).subs(minimum, simultaneous=True))

    work = []
    for sign in (-1, 1):
        state = dict(zip(forms, (-u / 2, u / 2, 0, 0)))
        # A uniform phase shift preserves all dynamics and sets the same
        # complete weighted mean, including the hidden fourth coordinate.
        state.update(
            dict(zip(phases, (-sign * alpha / 4,) * 3 + (3 * sign * alpha / 4,)))
        )
        observed = (r, delta, V, C, S, full)
        expected = (u, 0, 0, 1, 0, expected_storage)
        for actual, target in zip(observed, expected, strict=True):
            _zero(actual.subs(state, simultaneous=True) - target)
        for coordinates in (forms, phases):
            _zero(
                sum(a * b for a, b in zip(rho, coordinates)).subs(
                    state, simultaneous=True
                )
            )
        _zero(derivative(V).subs(state, simultaneous=True) + sign * v * s.sin(alpha))
        signed_work = derivative(K).subs(state, simultaneous=True)
        _zero(signed_work - sign * u * v * s.sin(alpha) / 2)
        _zero(derivative(rest).subs(state, simultaneous=True) + signed_work)
        work.append(signed_work)
    _zero(work[1] - work[0] - u * v * s.sin(alpha))
    # For 0<alpha<pi/2 and u>0 the signs differ strictly. The pair phase
    # starts at its reference with curvature 5/4. Smooth complete dynamics
    # and a sufficiently small neighborhood preserve a convex reference
    # chart locally; no response duration or trajectory is being asserted.
