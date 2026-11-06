"""Own-law algebra for the unchanged represented NUL class observation.

The fixed source builder is shared with the preceding constructed comparison;
no retained record is loaded. Alternative rows are independently declared here,
never obtained by relabeling a sine report. Endpoint controls are not a parameter
search: one amplitude bound covers the complete prospective epsilon interval.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.research.test_nul_causal_class_exit import (
    CRITICAL_EDGES,
    CRITICAL_GAP,
    HORIZON,
    MODEL,
    PERMUTATION,
    _reference_turn,
    _source_graph,
    _state,
)
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage
from tnfr.operators.factor_contracts import resolve_operator_factors
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research.nul_reset import bound_nul_capacity_reset
from tnfr.types import Glyph

PARAMETER = I(0, 3)
MAXIMUM_CURRENT = Q(4)


def _current(gap, epsilon):
    value = sin(I(gap))
    return value + epsilon * value**3


def _potential(gap, epsilon):
    value = cos(I(gap))
    return 1 - value + epsilon * (1 - value) ** 2 * (value + 2) / 3


def _primitive_rows(graph, epsilon):
    """Declared test-local full current and weighted phase rows in original t."""
    forms = tuple(Q(graph.nodes[i]["EPI"]) for i in graph)
    phases = tuple(Q(graph.nodes[i]["theta"]) for i in graph)
    capacities = tuple(Q(graph.nodes[i]["nu_f"]) for i in graph)
    gradients = tuple(
        sum(
            (Q(data["weight"]) * (forms[i] - forms[j]) for j, data in graph[i].items()),
            Q(0),
        )
        for i in graph
    )
    # Phase currents use bare incidences. Actual conductances enter q only.
    currents = tuple(
        sum((_current(phases[j] - phases[i], epsilon) for j in graph[i]), I(0))
        for i in graph
    )
    numerators = tuple(capacities[i] * gradients[i] / graph.degree[i] for i in graph)
    pi = pi_interval()
    form_rates = tuple(
        capacities[i] * currents[i] / (graph.degree[i] * pi) for i in graph
    )
    phase_rates = tuple(I(value) / pi for value in numerators)
    return gradients, currents, numerators, form_rates, phase_rates


@pytest.fixture(scope="module", autouse=True)
def no_event_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("cross-law controls must not execute or prepare a runtime event")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "execute_pointwise_stage",
            "execute_operator_major_stage",
            "_propose_pointwise",
        ):
            patch.setattr(network_stage, name, forbidden)
        yield


@pytest.fixture(scope="module")
def branches(no_event_execution):
    before = _source_graph()
    original = _state(before)
    factor = resolve_operator_factors(None, Glyph.NUL)["NUL_scale"]
    image = bound_nul_capacity_reset(
        factor=factor,
        capacity_bounds=(1.0,) * 3,
        form_bounds=(0.5,) * 3,
        pressure_bounds=(0.0,) * 3,
        epi_min=before.graph["EPI_MIN"],
        epi_max=before.graph["EPI_MAX"],
    )
    after = deepcopy(before)
    proposal = image.proposals[1]
    after.nodes[0].update(
        EPI=proposal.epi_after, nu_f=proposal.vf_after, delta_nfr=proposal.dnfr_after
    )
    assert _state(before) == original
    return before, after, image


@pytest.fixture(scope="module")
def incidence(branches):
    s = pytest.importorskip("sympy")
    graph = branches[0]
    edges = tuple(graph.edges)
    matrix = s.zeros(20, 45)
    for column, (left, right) in enumerate(edges):
        matrix[left, column], matrix[right, column] = 1, -1
    weights = s.diag(*(s.Rational(graph.edges[edge]["weight"]) for edge in edges))
    return s, edges, matrix, weights


def test_supplied_family_potential_current_and_uniform_parameter_bounds():
    s = pytest.importorskip("sympy")
    delta = s.Symbol("delta", real=True)
    epsilon = s.Symbol("epsilon", nonnegative=True)
    c, u = s.symbols("c u", real=True)
    potential = (
        1
        - s.cos(delta)
        + epsilon * (s.Rational(2, 3) - s.cos(delta) + s.cos(delta) ** 3 / 3)
    )
    current = s.sin(delta) + epsilon * s.sin(delta) ** 3
    assert s.trigsimp(s.diff(potential, delta) - current) == 0
    assert s.trigsimp(current.subs(delta, -delta) + current) == 0
    assert s.simplify(potential.subs(delta, -delta) - potential) == 0
    assert potential.subs(delta, 0) == 0
    assert s.diff(current, delta).subs(delta, 0) == 1
    assert (
        s.expand(
            1
            - c
            + epsilon * (s.Rational(2, 3) - c + c**3 / 3)
            - ((1 - c) + epsilon * (1 - c) ** 2 * (c + 2) / 3)
        )
        == 0
    )
    assert (
        s.trigsimp(
            s.diff(current, delta)
            - s.cos(delta) * (1 + 3 * epsilon * s.sin(delta) ** 2)
        )
        == 0
    )
    # These interval factors bound every cosine/sine and every epsilon in [0,3].
    cosine = I(-1, 1)
    nonnegative = (1 - cosine) + PARAMETER * (1 - cosine) ** 2 * (cosine + 2) / 3
    assert nonnegative.lo >= 0
    sine = I(-1, 1)
    assert (sine + PARAMETER * sine**3).abs_max <= MAXIMUM_CURRENT
    assert (I(-1, 1) * (1 + 3 * PARAMETER * I(0, 1))).abs_max <= 10
    assert s.expand((u + epsilon * u**3).subs(epsilon, 0) - u) == 0


def test_unchanged_source_and_actual_action_are_shared_across_current_laws(branches):
    before, after, image = branches
    assert len(before) == 20 and before.number_of_edges() == 45
    assert before.graph == after.graph
    assert tuple(before.edges(data="weight")) == tuple(after.edges(data="weight"))
    assert tuple(before.degree) == tuple(after.degree)
    assert all(before.nodes[i] == after.nodes[i] for i in range(1, 20))
    assert all(before.nodes[i]["theta"] == after.nodes[i]["theta"] for i in before)
    assert all(
        before.nodes[i]["delta_nfr"] == after.nodes[i]["delta_nfr"] == 0.0
        for i in before
    )
    assert image.image_form.nominal == image.factor / 2
    assert image.image_capacity.nominal == image.factor
    amplitude = (1 - image.factor) / 2
    assert Q(5, 128) < amplitude < Q(1, 16)
    assert HORIZON == Q(1, 16384)


@pytest.mark.parametrize("branch", (0, 1))
def test_exact_edge_algebra_preserves_own_weighted_energy_and_both_means(
    branches, incidence, branch
):
    s, _, matrix, weights = incidence
    graph = branches[branch]
    forms = s.Matrix(s.symbols("form0:20", real=True))
    edge_currents = s.Matrix(s.symbols("current0:45", real=True))
    gradient = matrix * weights * matrix.T * forms
    currents = matrix * edge_currents
    mobility = s.diag(
        *(s.Rational(graph.nodes[i]["nu_f"]) / graph.degree[i] for i in graph)
    )
    rho = s.Matrix(
        [graph.degree[i] / s.Rational(graph.nodes[i]["nu_f"]) for i in graph]
    )
    form_numerators = mobility * currents
    phase_numerators = mobility * gradient
    # U'=j makes the UNWEIGHTED phase gradient -B*j. Both original-time rows
    # have one common pi divisor, which may be removed from a zero identity.
    power = gradient.dot(form_numerators) - currents.dot(phase_numerators)
    assert s.expand(power) == 0
    assert s.expand(rho.dot(form_numerators)) == 0
    assert s.expand(rho.dot(phase_numerators)) == 0
    # Weighting the phase potential as well would change its gradient and fail
    # the selected bare-current identity on these actual nonunit contacts.
    wrong_phase_gradient = matrix * weights * edge_currents
    wrong_power = gradient.dot(form_numerators) - wrong_phase_gradient.dot(
        phase_numerators
    )
    assert s.expand(wrong_power) != 0


def test_exact_weighted_equivariance_applies_to_each_odd_current(branches, incidence):
    s, edges, matrix, weights = incidence
    node_map, edge_map = s.zeros(20), s.zeros(45)
    positions = {tuple(sorted(edge)): index for index, edge in enumerate(edges)}
    for node, destination in enumerate(PERMUTATION):
        node_map[destination, node] = 1
    for index, (left, right) in enumerate(edges):
        mapped = (PERMUTATION[left], PERMUTATION[right])
        destination = positions[tuple(sorted(mapped))]
        edge_map[destination, index] = 1 if mapped[0] < mapped[1] else -1
    assert node_map * matrix == matrix * edge_map
    assert edge_map * weights == weights * edge_map
    for graph in branches[:2]:
        mobility = s.diag(
            *(s.Rational(graph.nodes[i]["nu_f"]) / graph.degree[i] for i in graph)
        )
        assert node_map * mobility == mobility * node_map
        for key in ("EPI", "theta", "nu_f"):
            state = s.Matrix([s.Rational(graph.nodes[i][key]) for i in graph])
            assert node_map * state == state
    # Oddness commutes with the signed edge permutation, so these exact
    # identities intertwine BOTH full rows for every epsilon, not just sine.


@pytest.mark.parametrize("branch", (0, 1))
def test_zero_parameter_rows_match_the_shared_sine_owner_only_at_its_law(
    branches, branch
):
    graph = branches[branch]
    declared = _primitive_rows(graph, Q(0))
    sine = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    assert declared[0] == sine.form_gradient
    assert declared[2] == sine.phase_rate_numerators()
    for independent, shared in zip(
        declared[3] + declared[4], sine.form_rates + sine.phase_rates
    ):
        residual = independent - shared
        # The existing report and direct edge arithmetic have different
        # outward enclosure widths. Equality of their underlying zero-parameter
        # law is exact; no extra precision contract is imposed on either reader.
        assert residual.contains(0)


def test_distinct_current_changes_node_two_form_rate_without_a_common_clock_change(
    branches,
):
    after = branches[1]
    zero, endpoint = _primitive_rows(after, Q(0)), _primitive_rows(after, Q(3))
    pi = pi_interval()
    small, large = sin(I(Q(5, 4))), sin(I(CRITICAL_GAP))
    assert 0 < Q(5, 4) < CRITICAL_GAP < pi.lo / 2
    assert 0 < small.lo <= small.hi < large.lo
    coefficient = (small**3 - large**3) / (4 * pi)
    assert coefficient.hi < 0
    difference = endpoint[3][2] - zero[3][2]
    expected = 3 * coefficient
    assert difference.hi < 0
    assert max(difference.lo, expected.lo) <= min(difference.hi, expected.hi)
    # The node's four neighbors give -s^3-r_s^3+2*s^3 exactly. Linearity in
    # epsilon and the strictly negative coefficient distinguish every epsilon>0.
    assert endpoint[2] == zero[2] and endpoint[4] == zero[4]
    assert zero[4][2].lo > 0
    # The nonzero unchanged phase row forces any scalar clock multiplier at
    # this state to equal one, incompatible with the changed form row.


def _gap(graph, edge):
    left, right = edge
    return (
        I(Q(graph.nodes[right]["theta"]) - Q(graph.nodes[left]["theta"]))
        - 2 * _reference_turn(left, right) * pi_interval()
    )


def test_uniform_current_amplitude_closes_all_45_faces_for_every_parameter(branches):
    before, after, image = branches
    pi = pi_interval()
    # |j|<=M gives |x'|<=M/pi and |g''|<=4M/pi^2: the remainder scales
    # linearly with M, since the unchanged phase row differentiates form once.
    remainder = 2 * MAXIMUM_CURRENT * HORIZON**2 / pi.lo**2
    assert remainder < Q(4, 2**30)
    baseline_rates = _primitive_rows(before, Q(0))[2]
    after_rates = _primitive_rows(after, Q(0))[2]
    assert baseline_rates == (Q(0),) * 20
    crossed = set()
    for edge in before.edges:
        left, right = edge
        gap = _gap(before, edge)
        assert pi.lo / 2 - gap.abs_max - remainder > Q(4, 2**30)
        rate = I(after_rates[right] - after_rates[left]) / pi
        if edge in CRITICAL_EDGES:
            assert rate.lo > Q(5, 4096)
            assert gap.lo + rate.lo * HORIZON - remainder - pi.hi / 2 > Q(12, 2**30)
            crossed.add(edge)
        else:
            assert pi.lo / 2 - gap.abs_max - rate.abs_max * HORIZON - remainder > 0
    assert crossed == CRITICAL_EDGES
    amplitude = (1 - image.factor) / 2
    displacement = (
        2 * (amplitude * HORIZON + MAXIMUM_CURRENT * HORIZON**2 / pi.lo) / pi.lo
    )
    assert displacement < HORIZON / 24 + Q(4, 2**30) < Q(1, 2**18)
    for edge in after.edges:
        gap = _gap(after, edge)
        assert gap.abs_max + displacement < pi.lo
        if edge not in CRITICAL_EDGES:
            assert pi.lo / 2 - gap.abs_max - displacement > Q(1, 4) - Q(1, 2**18)
    for offset in (0, 10):
        cycle = tuple(offset + 2 * index for index in range(5))
        edges = tuple(zip(cycle, cycle[1:] + cycle[:1]))
        assert sum(_reference_turn(left, right) for left, right in edges) == -1
        for edge in edges:
            gap = _gap(after, edge)
            assert 0 < gap.lo - displacement <= gap.hi + displacement < pi.lo
    # Same finite observation for all epsilon does not assert equal trajectories
    # or exit times. The preceding coefficient proves different complete rows.


def test_positive_event_work_is_independent_of_the_changed_phase_potential(branches):
    before, after, image = branches
    amplitude = (1 - image.factor) / 2

    def form_storage(graph):
        return sum(
            (
                Q(data["weight"])
                * (Q(graph.nodes[i]["EPI"]) - Q(graph.nodes[j]["EPI"])) ** 2
                / 2
                for i, j, data in graph.edges(data=True)
            ),
            Q(0),
        )

    jump = form_storage(after) - form_storage(before)
    assert jump == Q(27, 8) * amplitude**2 > 0
    for epsilon in (Q(0), Q(3)):
        left = tuple(
            _potential(
                Q(before.nodes[j]["theta"]) - Q(before.nodes[i]["theta"]), epsilon
            )
            for i, j in before.edges
        )
        right = tuple(
            _potential(Q(after.nodes[j]["theta"]) - Q(after.nodes[i]["theta"]), epsilon)
            for i, j in after.edges
        )
        assert left == right
    # Identical primitive phases make this equality hold for every parameter.
    # Continuous branchwise conservation is not conservation across the reset.
