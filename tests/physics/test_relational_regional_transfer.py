"""Independent finite-graph identities for regional sine transfer.

These controls differentiate the complete fine law and use its graph storage.
They neither sample trajectories nor infer a hybrid event from a form rate.
"""

from dataclasses import replace
from fractions import Fraction as Q
from math import nextafter, pi

import mpmath as mp
import networkx as nx
import pytest

from tests.physics.test_relational_sine_replica import MODEL, PAIRS, _graph
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_sine_comparison as owner
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import relational_report_to_dict


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


@pytest.mark.parametrize("reader", ("regional", "increment"))
def test_accounting_exports_a_consistent_rebuilt_full_source(reader, monkeypatch):
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    captured = bound_relational_sine_exchange(graph, reference_model=MODEL)
    # Load the shared admission's lazy source types before replacing functions
    # that dependent modules may import by name on their first use.
    captured.phase_rate_numerators()
    supplied = replace(
        captured,
        epi=(0.25, -0.5, 0.75),
        phase=(0, 0.5, -0.25),
        capacity=(0.5, 2, 3),
        form_gradient=(Q(99),) * 3,
        relative_resultant=((I(99), I(99)),) * 3,
        form_rates=(I(99),) * 3,
        phase_rates=(I(99),) * 3,
        form_storage=Q(99),
        phase_storage=I(99),
        storage=I(198),
    )
    rebuild = owner._rebuild_sine_comparison
    calls = []

    def counted(source):
        calls.append(source)
        return rebuild(source)

    def forbidden(*args, **kwargs):
        pytest.fail("accounting must rebuild detached primitives, not recapture")

    monkeypatch.setattr(owner, "_rebuild_sine_comparison", counted)
    monkeypatch.setattr(owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(owner, "_stage", forbidden)
    if reader == "regional":
        report = supplied.regional_transfer(region=(0, 1))
        assert report.regional_weighted_form == 0
        assert report.complement_weighted_form == Q(1, 4)
    else:
        report = supplied.assess_form_increment(increments=(Q(1, 8), -Q(1, 4), 0))
        assert report.weighted_form_before == report.weighted_form_after == Q(1, 4)
        assert report.weighted_form_change == 0
        assert not report.endpoint_reachability_certified
    assert len(calls) == 1
    source = report.comparison
    assert source is not supplied
    assert source.epi == (Q(1, 4), -Q(1, 2), Q(3, 4))
    assert all(
        type(value) is Q for value in source.epi + source.phase + source.capacity
    )
    assert source.form_gradient == (Q(3, 4), -Q(2), Q(5, 4))
    assert source.form_storage == Q(17, 16)
    with mp.workdps(90):
        sine_half, sine_three_quarters = mp.sin(mp.mpf("0.5")), mp.sin(mp.mpf("0.75"))
        expected_rates = (
            sine_half / (2 * mp.pi),
            (-sine_half - sine_three_quarters) / mp.pi,
            3 * sine_three_quarters / mp.pi,
        )
        expected_phase_rates = (
            mp.mpf(3) / (8 * mp.pi),
            -2 / mp.pi,
            mp.mpf(15) / (4 * mp.pi),
        )
        expected_storage = (
            mp.mpf(17) / 16 + 2 - mp.cos(mp.mpf("0.5")) - mp.cos(mp.mpf("0.75"))
        )

        def contains(bound, value):
            low = mp.mpf(bound.lo.numerator) / bound.lo.denominator
            high = mp.mpf(bound.hi.numerator) / bound.hi.denominator
            assert low <= value <= high

        for bound, value in zip(source.form_rates, expected_rates):
            contains(bound, value)
        for bound, value in zip(source.phase_rates, expected_phase_rates):
            contains(bound, value)
        contains(source.storage, expected_storage)
        if reader == "regional":
            contains(report.regional_form_rate_bounds, -sine_three_quarters / mp.pi)
    assert report.to_dict()["report"] == relational_report_to_dict(report)["report"]
    assert supplied.form_rates == (I(99),) * 3
    assert supplied.form_storage == 99


def _fine_rows(s, edges, forms, phases, capacities, loss, phase_weight, beta):
    size = len(forms)
    adjacency = s.zeros(size)
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = 1
    degrees = tuple(sum(adjacency[i, j] for j in range(size)) for i in range(size))
    laplacian = s.diag(*degrees) - adjacency
    gradient = laplacian * s.Matrix(forms)
    currents = s.Matrix(
        [
            sum(adjacency[i, j] * s.sin(phases[j] - phases[i]) for j in range(size))
            for i in range(size)
        ]
    )
    mobility = s.diag(*(nu / degree for nu, degree in zip(capacities, degrees)))
    form_rates = mobility * (-loss * gradient + phase_weight * currents / s.pi)
    phase_rates = phase_weight * mobility * gradient / (beta * s.pi)
    return degrees, laplacian, form_rates, phase_rates


def test_cut_balance_cancels_internal_edges_in_both_complete_channels(symbolic):
    s = symbolic
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4))
    region = {0, 1, 2}
    forms = s.symbols("x0:6", real=True)
    phases = s.symbols("p0:6", real=True)
    capacities = s.symbols("nu0:6", positive=True)
    loss = s.symbols("e", nonnegative=True)
    weight, beta = s.symbols("w beta", positive=True)
    degrees, _, form_rates, phase_rates = _fine_rows(
        s, edges, forms, phases, capacities, loss, weight, beta
    )
    rho = tuple(degree / nu for degree, nu in zip(degrees, capacities))
    cut = tuple(
        (i, j) if i in region else (j, i)
        for i, j in edges
        if (i in region) != (j in region)
    )
    diffusion = sum(loss * (forms[j] - forms[i]) for i, j in cut)
    phase_current = sum(weight * s.sin(phases[j] - phases[i]) / s.pi for i, j in cut)
    regional_rate = sum(rho[i] * form_rates[i] for i in region)
    assert s.trigsimp(s.expand(regional_rate - diffusion - phase_current)) == 0
    assert s.trigsimp(s.expand(sum(rho[i] * form_rates[i] for i in range(6)))) == 0

    lifted_phase_rate = sum(rho[i] * phase_rates[i] for i in region)
    phase_boundary = weight * sum(forms[i] - forms[j] for i, j in cut) / (beta * s.pi)
    assert s.expand(lifted_phase_rate - phase_boundary) == 0
    assert s.expand(sum(rho[i] * phase_rates[i] for i in range(6))) == 0


def test_k22_full_rows_preserve_pairs_and_generate_phase_backreaction(symbolic):
    s = symbolic
    edges = ((0, 2), (0, 3), (1, 2), (1, 3))
    forms = s.symbols("x0:4", real=True)
    phases = s.symbols("p0:4", real=True)
    nu, weight, beta = s.symbols("nu w beta", positive=True)
    xa, xb, pa, pb, common = s.symbols("XA XB PA PB C", real=True)
    source = dict(zip(forms + phases, (xa, xa, xb, xb, pa, pa, pb, pb)))
    _, _, form_rates, phase_rates = _fine_rows(
        s, edges, forms, phases, (nu,) * 4, 0, weight, beta
    )
    captured_form = form_rates.subs(source).applyfunc(s.simplify)
    captured_phase = phase_rates.subs(source).applyfunc(s.simplify)
    assert captured_form[0] == captured_form[1]
    assert captured_form[2] == captured_form[3] == -captured_form[0]
    assert captured_phase[0] == captured_phase[1]
    assert captured_phase[2] == captured_phase[3]
    assert s.expand(captured_phase[2] + captured_phase[0]) == 0
    assert s.trigsimp(captured_form[0] - weight * nu * s.sin(pb - pa) / s.pi) == 0

    state = s.Matrix(forms + phases)
    fine_field = form_rates.col_join(phase_rates)
    acceleration = phase_rates.jacobian(state) * fine_field
    # Simultaneous substitution would leave the symbolic block means; apply
    # the fine synchronization first, then the equal-form preparation.
    initial_velocity = captured_phase.subs({xa: common, xb: common})
    initial_acceleration = acceleration.subs(source).subs({xa: common, xb: common})
    assert initial_velocity == s.zeros(4, 1)
    expected = 2 * weight**2 * nu**2 * s.sin(pb - pa) / (beta * s.pi**2)
    assert all(s.trigsimp(initial_acceleration[i] - expected) == 0 for i in (0, 1))
    assert all(s.trigsimp(initial_acceleration[i] + expected) == 0 for i in (2, 3))
    assert s.trigsimp(captured_form[0].subs(pb, pa)) == 0
    assert s.trigsimp(expected.subs(pb, 2 * pa - pb) + expected) == 0


def test_inherited_energy_sets_finite_transfer_without_rate_fitting(symbolic):
    s = symbolic
    edges = ((0, 2), (0, 3), (1, 2), (1, 3))
    xa, xb, delta, initial_delta = s.symbols("XA XB delta delta0", real=True)
    nu, weight, beta = s.symbols("nu w beta", positive=True)
    forms = (xa, xa, xb, xb)
    phases = (0, 0, delta, delta)
    _, laplacian, form_rates, phase_rates = _fine_rows(
        s, edges, forms, phases, (nu,) * 4, 0, weight, beta
    )
    fine_form = (s.Matrix(forms).T * laplacian * s.Matrix(forms))[0] / 2
    fine_phase = beta * sum(1 - s.cos(phases[j] - phases[i]) for i, j in edges)
    fine_energy = s.expand(fine_form + fine_phase)
    coarse_energy = (xa - xb) ** 2 / 2 + beta * (1 - s.cos(delta))
    assert s.expand(fine_energy - 4 * coarse_energy) == 0

    delta_rate = phase_rates[2] - phase_rates[0]
    energy_rate = (
        s.diff(coarse_energy, xa) * form_rates[0]
        + s.diff(coarse_energy, xb) * form_rates[2]
        + s.diff(coarse_energy, delta) * delta_rate
    )
    assert s.simplify(energy_rate) == 0
    relative_acceleration = (
        s.diff(delta_rate, xa) * form_rates[0] + s.diff(delta_rate, xb) * form_rates[2]
    )
    assert (
        s.simplify(
            relative_acceleration
            + 4 * weight**2 * nu**2 * s.sin(delta) / (beta * s.pi**2)
        )
        == 0
    )

    initial_energy = coarse_energy.subs(xb, xa).subs(delta, initial_delta)
    predicted_squared_block_gain = beta * s.sin(initial_delta / 2) ** 2
    # A block displacement g means D=2g; at the attained phase crossing
    # the inherited P2 energy is 2g^2. Libration supplies attainability.
    assert s.trigsimp(2 * predicted_squared_block_gain - initial_energy) == 0


def test_positive_al_increment_violates_mean_even_when_storage_decreases(symbolic):
    s = symbolic
    # A unit edge already suffices for this full-state endpoint obstruction.
    # Initially x=(-1,0). Boost its first form by b in (0,1), reducing the
    # edge difference and storage while increasing the conserved form sum.
    boost, nu = s.symbols("b nu", positive=True)
    before = s.Matrix((-1, 0))
    jump = s.Matrix((boost, 0))
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    storage_jump = s.expand(
        ((before + jump).T * laplacian * (before + jump))[0] / 2
        - (before.T * laplacian * before)[0] / 2
    )
    weighted_form_jump = sum(jump) / nu
    assert s.factor(storage_jump) == boost * (boost - 2) / 2
    assert weighted_form_jump.is_positive
    assert storage_jump.subs(boost, s.Rational(1, 2)) < 0
    assert weighted_form_jump.subs(boost, 0) == 0
    assert storage_jump.subs(boost, 0) == 0


def test_antipodal_pair_restores_form_current_through_the_complete_fine_field(symbolic):
    s = symbolic
    graph = _graph()
    forms, phases = s.symbols("x0:10", real=True), s.symbols("p0:10", real=True)
    u = s.Symbol("u", real=True)
    degrees, laplacian, form_rates, phase_rates = _fine_rows(
        s, tuple(graph.edges()), forms, phases, (s.Integer(1),) * 10, 0, 1, 1
    )
    assert degrees == (4,) * 10
    source = dict(zip(forms + phases, (u, -u) + (0,) * 8 + (0, s.pi) + (0,) * 8))
    field = form_rates.col_join(phase_rates)
    captured = field.subs(source)
    assert captured[:10, :] == s.zeros(10, 1)
    assert captured[10:, :] == s.Matrix((u / s.pi, -u / s.pi) + (0,) * 8)
    source_forms = s.Matrix(forms).subs(source)
    assert s.expand((source_forms.T * laplacian * source_forms)[0] / 2) == 4 * u**2
    phase_storage = sum(1 - s.cos(phases[j] - phases[i]) for i, j in graph.edges())
    assert phase_storage.subs(source) == 8

    # Differentiate the full field before evaluation; the resultant angle is
    # undefined and the strict circular midpoint chart is unavailable here.
    state = s.Matrix(forms + phases)
    acceleration = (field.jacobian(state).subs(source) * captured).applyfunc(s.simplify)
    assert acceleration[10:, :] == s.zeros(10, 1)
    half = s.Rational(1, 2)
    assert acceleration[:10, :] == (u / s.pi**2) * s.Matrix(
        (-1, -1, half, half, 0, 0, 0, 0, half, half)
    )
    means = tuple(s.simplify((acceleration[i] + acceleration[j]) / 2) for i, j in PAIRS)
    assert means == (
        -u / s.pi**2,
        u / (2 * s.pi**2),
        0,
        0,
        u / (2 * s.pi**2),
    )
    assert s.simplify(sum(8 * value for value in means)) == 0

    za = (s.exp(s.I * phases[0]) + s.exp(s.I * phases[1])) / 2
    assert s.simplify(za.subs(source)) == 0
    za_rate = sum(s.diff(za, phases[i]) * phase_rates[i] for i in (0, 1))
    form_phase_moment = (
        forms[0] * s.exp(s.I * phases[0]) + forms[1] * s.exp(s.I * phases[1])
    ) / 2
    neighbor_mean = sum(forms[i] for pair in (PAIRS[1], PAIRS[-1]) for i in pair) / 4
    assert (
        s.expand(za_rate - s.I * (form_phase_moment - neighbor_mean * za) / s.pi) == 0
    )
    assert s.simplify(za_rate.subs(source)) == s.I * u / s.pi
    for amplitude in (s.Rational(1, 8), -s.Rational(1, 8)):
        assert s.simplify(means[0].subs(u, amplitude)).is_negative == (amplitude > 0)
    assert captured.subs(u, 0) == s.zeros(20, 1)

    # Zero form current does not remove the phase channel: changing the
    # neighboring pair's common form changes the primitive mean phase rate.
    mean_phase_rate = (phase_rates[0] + phase_rates[1]) / 2
    neighboring_response = sum(s.diff(mean_phase_rate, forms[i]) for i in PAIRS[1])
    assert neighboring_response == -1 / (2 * s.pi)


def test_pair_phasor_current_uses_original_degree_and_weighted_boundary(symbolic):
    s = symbolic
    first, second, third, fourth = s.symbols("a b c d", real=True)
    za = (s.exp(s.I * first) + s.exp(s.I * second)) / 2
    zb = (s.exp(s.I * third) + s.exp(s.I * fourth)) / 2
    cut = sum(
        s.sin(outside - inside)
        for inside in (first, second)
        for outside in (third, fourth)
    )
    phasor_current = s.im(s.conjugate(za) * zb).expand(complex=True)
    assert s.trigsimp(cut - 4 * phasor_current) == 0
    # Fine degree four, two receiving nodes, and weights d/nu=4 give
    # distinct block-mean and weighted-regional normalization factors.
    mean_contribution = cut / (8 * s.pi)
    weighted_contribution = cut / s.pi
    assert s.trigsimp(mean_contribution - phasor_current / (2 * s.pi)) == 0
    assert s.simplify(weighted_contribution - 8 * mean_contribution) == 0


def test_single_port_reflection_retains_the_full_nonstationary_probe_response():
    graph = nx.cycle_graph(5)
    graph.add_edges_from(((0, 5), (5, 6)))
    # These exact dyadic phases are near, not equal to, an ideal C5 twist.
    # Reflection compares the actual captured representatives without using
    # a rounded multiple of 2*pi as an exact circular identity.
    angle = Q(5, 4)
    phases = (0, angle, 2 * angle, -2 * angle, -angle, Q(1, 4), -Q(3, 8))
    forms = (Q(1, 8), 0, 0, 0, 0, Q(3, 16), -Q(1, 2))
    capacities = (1, 1, 1, 1, 1, 2, Q(1, 2))
    for node, form, phase, capacity in zip(graph, forms, phases, capacities):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    reflection = (0, 4, 3, 2, 1, 5, 6)
    reflected = graph.copy()
    for node in graph:
        reflected.nodes[node].update(graph.nodes[reflection[node]])
    before = tuple(dict(node_data) for _, node_data in graph.nodes(data=True))
    reflected_before = tuple(
        dict(node_data) for _, node_data in reflected.nodes(data=True)
    )

    first = bound_relational_sine_exchange(graph, reference_model=MODEL)
    second = bound_relational_sine_exchange(reflected, reference_model=MODEL)
    assert second.phase == tuple(first.phase[node] for node in reflection)
    assert second.form_rates == tuple(first.form_rates[node] for node in reflection)
    assert second.phase_rates == tuple(first.phase_rates[node] for node in reflection)
    assert second.storage == first.storage
    first_probe = first.regional_transfer(region=(5, 6))
    second_probe = second.regional_transfer(region=(5, 6))
    assert (
        first_probe.regional_form_rate_bounds == second_probe.regional_form_rate_bounds
    )
    # The null is the difference between handed preparations, not a frozen
    # environment or zero current. Its internal edge cancels from the ledger.
    current = first_probe.regional_form_rate_bounds
    assert current.hi < 0
    with mp.workdps(90):
        expected = -mp.sin(mp.mpf(1) / 4) / mp.pi
        assert mp.mpf(current.lo.numerator) / current.lo.denominator <= expected
        assert expected <= mp.mpf(current.hi.numerator) / current.hi.denominator
    assert tuple(dict(data) for _, data in graph.nodes(data=True)) == before
    assert (
        tuple(dict(data) for _, data in reflected.nodes(data=True)) == reflected_before
    )


def test_two_ordered_probe_contacts_resolve_orientation_at_equal_storage():
    graphs, sources, probes = [], [], []
    angle = 2 * pi / 5
    for sign in (1, -1):
        graph = nx.cycle_graph(5)
        graph.add_edges_from(((1, 5), (4, 6)))
        phases = (
            0,
            sign * angle,
            sign * 2 * angle,
            -sign * 2 * angle,
            -sign * angle,
            0,
            0,
        )
        for node, phase in zip(graph, phases):
            graph.nodes[node].update(EPI=0, theta=phase, nu_f=1)
        before = tuple(dict(data) for _, data in graph.nodes(data=True))
        source = bound_relational_sine_exchange(graph, reference_model=MODEL)
        left = source.regional_transfer(region=(5,))
        right = source.regional_transfer(region=(6,))
        assert left.form_weights[5] == right.form_weights[6] == 1
        assert left.regional_form_rate_bounds == -right.regional_form_rate_bounds
        assert source.phase_rate_numerators() == (Q(0),) * 7
        graphs.append((graph, before))
        sources.append(source)
        probes.append(left.regional_form_rate_bounds - right.regional_form_rate_bounds)

    positive, negative = sources
    assert positive.form_storage == negative.form_storage == 0
    assert positive.storage == negative.storage
    assert probes[0] == -probes[1]
    assert probes[0].lo > 0
    assert positive.phase[1] == Q(angle)
    # Independently evaluate the represented angle, not mathematical 2*pi/5.
    # Its closing cycle gap has cosine cos(4*angle), so the ideal simplification
    # H=7*(1-cos(2*pi/5)) is deliberately not used for this captured state.
    with mp.workdps(90):
        represented = mp.mpf(Q(angle).numerator) / Q(angle).denominator
        expected_current = 2 * mp.sin(represented) / mp.pi
        expected_storage = 6 * (1 - mp.cos(represented)) + 1 - mp.cos(4 * represented)
        assert represented != 2 * mp.pi / 5
        for bound, expected in (
            (probes[0], expected_current),
            (positive.storage, expected_storage),
        ):
            assert mp.mpf(bound.lo.numerator) / bound.lo.denominator <= expected
            assert expected <= mp.mpf(bound.hi.numerator) / bound.hi.denominator
    for graph, before in graphs:
        assert tuple(dict(data) for _, data in graph.nodes(data=True)) == before


@pytest.mark.parametrize("phase", (pi, nextafter(pi, float("inf"))))
def test_represented_antipodal_neighbors_keep_current_and_acceleration_distinct(phase):
    # Neither represented phase equals mathematical pi. The same primitive
    # support remains admitted on both sides; no midpoint angle is requested.
    reports = []
    for u in (Q(1, 8), Q(-1, 8), Q(0)):
        graph = _graph(phase=(0,) * 5, internal_form=(u, 0, 0, 0, 0))
        graph.nodes[1]["theta"] = phase
        source = bound_relational_sine_exchange(graph, reference_model=MODEL)
        transfer = source.regional_transfer(region=PAIRS[0])
        kinematics = source.resultant_kinematics()
        form_acceleration = tuple(
            imaginary / (degree * pi_interval())
            for (_, imaginary), degree in zip(
                kinematics.resultant_rate_bounds, source.degrees
            )
        )
        assert source.phase[1] == Q(phase)
        assert kinematics.phase_rate_numerators == (u, -u) + (Q(0),) * 8
        assert not transfer.regional_form_rate_bounds.contains(0)
        assert source.form_rates[1] != I(0)

        with mp.workdps(100):
            angle = mp.mpf(Q(phase).numerator) / Q(phase).denominator
            amplitude = mp.mpf(u.numerator) / u.denominator
            sine, cosine = mp.sin(angle), mp.cos(angle)
            expected_current = -4 * sine / mp.pi
            expected_acceleration = (
                (
                    -amplitude / mp.pi**2,
                    amplitude * cosine / mp.pi**2,
                )
                + (amplitude * (1 - cosine) / (4 * mp.pi**2),) * 2
                + (mp.mpf(0),) * 4
                + (amplitude * (1 - cosine) / (4 * mp.pi**2),) * 2
            )

            def contains(bound, value):
                lower = mp.mpf(bound.lo.numerator) / bound.lo.denominator
                upper = mp.mpf(bound.hi.numerator) / bound.hi.denominator
                assert lower <= value <= upper

            contains(transfer.regional_form_rate_bounds, expected_current)
            for bound, expected in zip(form_acceleration, expected_acceleration):
                contains(bound, expected)
        reports.append((source, transfer, form_acceleration))

    positive, negative, zero = reports
    assert positive[0].form_rates == negative[0].form_rates == zero[0].form_rates
    assert (
        positive[1].regional_form_rate_bounds == negative[1].regional_form_rate_bounds
    )
    assert all(first == -second for first, second in zip(positive[2], negative[2]))
    assert zero[2] == (I(0),) * 10
    # u=0 is stationary at the exact antipode in the symbolic control above;
    # it is not stationary at either represented phase used in this reader.
    assert any(value != I(0) for value in zero[0].form_rates)
