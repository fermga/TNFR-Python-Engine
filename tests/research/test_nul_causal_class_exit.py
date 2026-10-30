"""Fixed represented same-source NUL/no-event class-exit comparison.

Only read-only admission, a pure numerical image and instantaneous shared-law
arithmetic are evaluated. Finite-time conclusions use the independently proved
global remainder bound; no event, integration, retained source or search occurs.
"""

from collections import deque
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.config.defaults_core import CORE_DEFAULTS
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import network_stage
from tnfr.operators.definitions import Coherence, Contraction, Silence
from tnfr.operators.factor_contracts import resolve_operator_factors
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.preconditions import validate_contraction
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution
from tnfr.research.nul_reset import bound_nul_capacity_reset
from tnfr.types import Glyph

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
PERMUTATION = (0, 1) + tuple(i + 1 if i % 2 == 0 else i - 1 for i in range(2, 20))
PHASE_ONE = Q(-5382069, 2**24)
CRITICAL_GAP = Q(26353589, 2**24)
CRITICAL_EDGES = {(1, 2), (1, 3)}
HORIZON = Q(1, 2**14)


def _source_graph():
    graph = nx.Graph(
        RANDOM_SEED=17,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        CLIP_MODE="hard",
        EPI_MIN=CORE_DEFAULTS["EPI_MIN"],
        EPI_MAX=CORE_DEFAULTS["EPI_MAX"],
        NUL_MIN_VF=0.1,
        NUL_MIN_EPI=0.1,
        NUL_MAX_DENSITY=10.0,
        COLLECT_OPERATOR_METRICS=False,
        VALIDATE_NODAL_EQUATION=False,
    )
    for node in range(20):
        history = (
            ("AL", "UM", "SHA", "IL", "OZ")
            if node == 0
            else (
                ("AL", "AL", "AL", "UM", "SHA", "IL", "OZ")
                if node == 1
                else ("AL", "IL")
            )
        )
        graph.add_node(
            node,
            EPI=0.5,
            nu_f=1.0,
            delta_nfr=0.0,
            theta=float(PHASE_ONE) if node == 1 else 5 * (node % 10 // 2) / 4,
            glyph_history=deque(history, maxlen=8),
            _grammar_u2_debt=1 if node < 2 else 0,
            _grammar_prior_coherence=True,
            _operator_step=len(history),
        )
    for i in range(20):
        for j in range(i + 1, 20):
            if i // 10 == j // 10 and (i % 10 // 2 - j % 10 // 2) % 5 in (1, 4):
                graph.add_edge(i, j, weight=1.0, length=1.0)
    for i, j, weight in (
        (0, 1, 7 / 8),
        (0, 10, 15 / 16),
        (0, 11, 15 / 16),
        (1, 10, 13 / 16),
        (1, 11, 13 / 16),
    ):
        graph.add_edge(i, j, weight=weight, length=weight)
    return graph


def _field(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _state(graph):
    return (
        deepcopy(graph.graph),
        tuple(
            (node, deepcopy(data), tuple(graph[node]))
            for node, data in graph.nodes(data=True)
        ),
        tuple((i, j, deepcopy(data)) for i, j, data in graph.edges(data=True)),
    )


@pytest.fixture(scope="module", autouse=True)
def no_event_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("causal controls must not execute or prepare a runtime event")

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
    source = _source_graph()
    original = _state(source)
    baseline = deepcopy(source)
    factor = resolve_operator_factors(None, Glyph.NUL)["NUL_scale"]
    image = bound_nul_capacity_reset(
        factor=factor,
        capacity_bounds=(1.0,) * 3,
        form_bounds=(0.5,) * 3,
        pressure_bounds=(0.0,) * 3,
        epi_min=source.graph["EPI_MIN"],
        epi_max=source.graph["EPI_MAX"],
    )
    proposal = image.proposals[1]
    successor = deepcopy(source)
    successor.nodes[0].update(
        EPI=proposal.epi_after,
        nu_f=proposal.vf_after,
        delta_nfr=proposal.dnfr_after,
    )
    # This is the mathematical input image, not a runtime commit. History,
    # debt and telemetry are unchanged because no event is executed here.
    assert _state(source) == _state(baseline) == original
    return source, baseline, successor, image, _field(baseline), _field(successor)


def _reference_turn(left, right):
    difference = right % 10 // 2 - left % 10 // 2
    return 1 if difference == 4 else -1 if difference == -4 else 0


def _gap(source, edge):
    left, right = edge
    return (
        I(source.phase[right] - source.phase[left])
        - 2 * _reference_turn(left, right) * pi_interval()
    )


def _independent_gradient(graph, node):
    form = Q(graph.nodes[node]["EPI"])
    return sum(
        (
            Q(data["weight"]) * (form - Q(graph.nodes[neighbor]["EPI"]))
            for neighbor, data in graph[node].items()
        ),
        Q(0),
    )


def test_matched_branches_retain_exact_source_and_only_actual_target_image(branches):
    source, baseline, successor, image, before, after = branches
    factor = image.factor
    assert len(source) == 20 and source.number_of_edges() == 45
    assert before.nodes == after.nodes == tuple(range(20))
    assert before.edges == after.edges
    assert before.conductances == after.conductances
    assert before.degrees == after.degrees == (7, 7) + (4,) * 8 + (6, 6) + (4,) * 8
    assert _state(source) == _state(baseline)
    assert source.graph == successor.graph
    assert all(source.nodes[node] == successor.nodes[node] for node in range(1, 20))
    assert before.epi == (Q(1, 2),) * 20 and before.capacity == (Q(1),) * 20
    assert after.epi == (factor / 2,) + before.epi[1:]
    assert after.capacity == (factor,) + before.capacity[1:]
    assert before.phase == after.phase
    assert before.phase[1] == PHASE_ONE
    assert Q(source.nodes[1]["theta"]) == PHASE_ONE
    assert image.image_form.nominal == factor / 2
    assert image.image_capacity.nominal == factor
    assert image.image_stored_pressure.nominal == 0
    assert image.proposals[1].clip_delta == 0.0
    assert (source.graph["EPI_MIN"], source.graph["EPI_MAX"]) == (-1.0, 1.0)
    amplitude = (1 - factor) / 2
    assert Q(5, 128) < amplitude < Q(1, 16)
    assert max(after.epi) - min(after.epi) == amplitude


def test_original_context_admits_singleton_stage_without_mutating_either_branch(
    branches,
):
    source, baseline, _, _, _, _ = branches
    original = _state(source)
    operator = Contraction()
    context = ValidatedSequence(
        (operator, Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    ).step(0)
    validate_contraction(source, 0)
    preflight = network_stage._preflight_two_phase_stage(
        source,
        operator,
        (0,),
        sequence_context=context,
        execution_kwargs={},
    )
    assert preflight.exact_stage
    assert preflight.snapshot is not source
    assert _state(preflight.snapshot) == original
    assert _state(source) == _state(baseline) == original
    assert source.nodes[0]["_grammar_u2_debt"] == 1
    assert not hasattr(source, "_last_operator_applied")


def test_both_successor_laws_keep_exact_weighted_twin_symmetry(branches):
    _, _, _, _, before, after = branches
    for source in (before, after):
        symmetry = assess_sine_support_involution(
            source, permutation_indices=PERMUTATION
        )
        assert symmetry.source_trajectory_reduction_certified
        assert symmetry.orbits == ((0,), (1,)) + tuple(
            (i, i + 1) for i in range(2, 20, 2)
        )
        assert symmetry.multiplicities == (1, 1) + (2,) * 9
        assert all(0 < capacity <= 1 for capacity in source.capacity)
        assert all(0 < weight <= 1 for weight in source.conductances)


def test_every_source_face_and_represented_critical_margin_are_admitted(branches):
    before, pi = branches[4], pi_interval()
    found = set()
    for edge in before.edges:
        gap = _gap(before, edge)
        margin = pi.lo / 2 - gap.abs_max
        if edge in CRITICAL_EDGES:
            assert gap.lo == gap.hi == CRITICAL_GAP
            uncertainty = pi / 2 - CRITICAL_GAP
            assert Q(1, 2**27) < uncertainty.lo <= uncertainty.hi < Q(1, 2**24)
            found.add(edge)
        else:
            assert margin > Q(1, 4)
        assert margin > 0
    assert len(before.edges) == 45 and found == CRITICAL_EDGES


def test_shared_full_law_initial_rates_match_independent_neighbor_gradients(branches):
    _, baseline, successor, image, before, after = branches
    amplitude = (1 - image.factor) / 2
    for graph, source in ((baseline, before), (successor, after)):
        exact = tuple(_independent_gradient(graph, node) for node in source.nodes)
        assert source.form_gradient == exact
        expected = tuple(
            capacity * gradient / degree
            for capacity, gradient, degree in zip(
                source.capacity, exact, source.degrees, strict=True
            )
        )
        assert source.phase_rate_numerators() == expected
    assert before.form_gradient == (Q(0),) * 20
    assert before.phase_rate_numerators() == (Q(0),) * 20
    # The baseline is not stationary: unequal phases immediately change form,
    # so its future phase velocity cannot be set to zero on the whole interval.
    assert any(not rate.contains(0) for rate in before.form_rates)
    assert any(not pressure.contains(0) for pressure in before.pressure)
    assert all(baseline.nodes[node]["delta_nfr"] == 0.0 for node in baseline)
    assert after.form_gradient[2] == after.form_gradient[3] == amplitude
    assert after.form_gradient[1] == Q(7, 8) * amplitude
    rates = after.phase_rate_numerators()
    assert rates[2] - rates[1] == rates[3] - rates[1] == amplitude / 8
    initial_rate = I(amplitude / 8) / pi_interval()
    assert initial_rate.lo > Q(5, 4096)


def test_global_full_law_remainder_gives_matched_finite_time_class_exit(branches):
    before, after, pi = branches[4], branches[5], pi_interval()
    # Globally |x_i'|<=1/pi; consequently |q_i'|<=2*d_i/pi,
    # |theta_i''|<=2/pi^2 and |g_e''|<=4/pi^2 in both held laws.
    # This bounds the complete Taylor remainder without an integrated response.
    remainder = 2 * HORIZON**2 / pi.lo**2
    assert remainder < Q(1, 2**30)
    baseline_rates = before.phase_rate_numerators()
    after_rates = after.phase_rate_numerators()
    baseline_margins, crossed = [], set()
    for edge in before.edges:
        left, right = edge
        gap = _gap(before, edge)
        assert baseline_rates[right] - baseline_rates[left] == 0
        baseline_margin = pi.lo / 2 - gap.abs_max - remainder
        assert baseline_margin > Q(7, 2**30)
        baseline_margins.append(baseline_margin)
        initial_rate = I(after_rates[right] - after_rates[left]) / pi
        if edge in CRITICAL_EDGES:
            overshoot = gap.lo + initial_rate.lo * HORIZON - remainder - pi.hi / 2
            assert overshoot > Q(15, 2**30)
            crossed.add(edge)
        else:
            whole_window_margin = (
                pi.lo / 2 - gap.abs_max - initial_rate.abs_max * HORIZON - remainder
            )
            assert whole_window_margin > 0
    assert len(baseline_margins) == 45 and crossed == CRITICAL_EDGES
    # Strict positive overshoot at T gives an earlier acute exit by continuity;
    # the matched no-event law stays inside on the entire closed [0,T].


def test_other_faces_and_principal_cuts_keep_the_original_windings(branches):
    after, pi = branches[5], pi_interval()
    amplitude = max(after.epi) - min(after.epi)
    # The global diameter bound also controls every edge away from its acute
    # face: B(t)=2/pi*(A*t+t^2/pi), increasing throughout this fixed window.
    displacement = 2 * (amplitude * HORIZON + HORIZON**2 / pi.lo) / pi.lo
    assert displacement < HORIZON / 24 + Q(1, 2**30) < Q(1, 2**18)
    for edge in after.edges:
        gap = _gap(after, edge)
        assert gap.abs_max + displacement < pi.lo
        if edge not in CRITICAL_EDGES:
            assert pi.lo / 2 - gap.abs_max - displacement > Q(1, 4) - Q(1, 2**18)
    for offset in (0, 10):
        cycle = tuple(offset + 2 * index for index in range(5))
        edges = tuple(zip(cycle, cycle[1:] + cycle[:1]))
        assert sum(_reference_turn(left, right) for left, right in edges) == -1
        assert (
            sum((after.phase[right] - after.phase[left] for left, right in edges), Q(0))
            == 0
        )
        for edge in edges:
            gap = _gap(after, edge)
            assert gap.lo - displacement > 0
            assert gap.hi + displacement < pi.lo
    # The same retained principal branches telescope to 2*pi on each cycle.
    # Exiting an acute face at pi/2 therefore does not imply a winding change.
