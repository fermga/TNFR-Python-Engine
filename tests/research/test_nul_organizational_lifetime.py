"""Constructed controls for the frozen marked-contact lifetime argument.

No retained preparation, event execution, trajectory or integration is used.
The representative coupled endpoints below exercise shared owners; the
canonical analytical argument, not endpoint sampling, covers the whole family.
"""

import math
from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

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
FORM = (19 / 64, 5 / 16, 21 / 64)
PORT_CAPACITY = (15 / 32, 1 / 2, 17 / 32)
OTHER_CAPACITY = (23 / 32, 3 / 4, 25 / 32)
PRESSURE = (-0.5, 0.0, 0.5)
HORIZON = Q(1, 4)
DIAMETER = Q(5, 32)
MARGIN = Q(1, 20)


def _factor():
    return resolve_operator_factors(None, Glyph.NUL)["NUL_scale"]


def _cell(node):
    return node if node < 2 else node // 2 + 1


def _graph(variant=0):
    """Build one coupled member independently of every retained contact file."""
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
        choice = (_cell(node) + variant) % 3
        phase = 5 * (node % 10 // 2) / 4 + (choice - 1) / 32
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
            EPI=FORM[choice],
            theta=phase,
            nu_f=(PORT_CAPACITY if node < 2 else OTHER_CAPACITY)[choice],
            delta_nfr=PRESSURE[choice],
            glyph_history=deque(history, maxlen=8),
            _grammar_u2_debt=1 if node < 2 else 0,
            _grammar_prior_coherence=True,
            _operator_step=len(history),
        )
    for i in range(20):
        for j in range(i + 1, 20):
            if i // 10 == j // 10 and (i % 10 // 2 - j % 10 // 2) % 5 in (1, 4):
                graph.add_edge(i, j, weight=1.0, length=1.0)
    for orbit, edges in enumerate((((0, 1),), ((0, 10), (0, 11)), ((1, 10), (1, 11)))):
        weight = 7 / 8 + ((orbit + variant) % 3 - 1) / 32
        for edge in edges:
            graph.add_edge(*edge, weight=weight, length=weight)
    return graph


def _field(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _symmetry(graph):
    return assess_sine_support_involution(
        _field(graph), permutation_indices=PERMUTATION
    )


def _image(*, capacity=PORT_CAPACITY, form=FORM, pressure=PRESSURE):
    return bound_nul_capacity_reset(
        factor=_factor(),
        capacity_bounds=capacity,
        form_bounds=form,
        pressure_bounds=pressure,
        epi_min=CORE_DEFAULTS["EPI_MIN"],
        epi_max=CORE_DEFAULTS["EPI_MAX"],
    )


def _detached_successor(graph, target=0):
    """Construct a pure numerical-image record, not an executed graph event."""
    node = graph.nodes[target]
    image = _image(
        capacity=(node["nu_f"],) * 3,
        form=(node["EPI"],) * 3,
        pressure=(node["delta_nfr"],) * 3,
    )
    proposal = image.proposals[1]
    after = deepcopy(graph)
    after.nodes[target].update(
        EPI=proposal.epi_after,
        nu_f=proposal.vf_after,
        delta_nfr=proposal.dnfr_after,
    )
    return after


@pytest.fixture(autouse=True)
def no_event_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("lifetime controls must not execute or prepare a runtime event")

    for name in (
        "execute_pointwise_stage",
        "execute_operator_major_stage",
        "_propose_pointwise",
    ):
        monkeypatch.setattr(network_stage, name, forbidden)


def test_registered_default_and_actual_joint_image_fit_frozen_bounds():
    image = _image()
    assert Q(7, 8) < Q(_factor()) < 1
    assert (CORE_DEFAULTS["EPI_MIN"], CORE_DEFAULTS["EPI_MAX"]) == (-1.0, 1.0)
    assert image.image_capacity.lower > Q(105, 256)
    assert image.image_capacity.upper < Q(17, 32)
    assert image.image_form.lower > Q(133, 512)
    assert image.image_form.upper < Q(21, 64)
    assert Q(21, 64) - image.image_form.lower < DIAMETER
    assert image.image_stored_pressure.lower > -Q(4, 7)
    assert image.image_stored_pressure.upper < Q(4, 7)
    assert all(proposal.clip_delta == 0.0 for proposal in image.proposals)
    # Ratios use represented products, not an assumed ideal factor equality.
    for ratio, proposal in zip(image.endpoint_capacity_ratios, image.proposals):
        assert ratio == Q(proposal.vf_after) / Q(proposal.vf_before)
    interior = _image(
        capacity=(PORT_CAPACITY[0], math.nextafter(0.5, math.inf), PORT_CAPACITY[2])
    )
    assert interior.source_capacity.nominal < image.source_capacity.upper
    assert (
        interior.image_capacity.nominal
        != interior.factor * interior.source_capacity.nominal
    )


@pytest.mark.parametrize("variant", range(3))
def test_coupled_members_and_singleton_image_preserve_actual_weighted_involution(
    variant,
):
    graph = _graph(variant)
    before = deepcopy(graph)
    after = _detached_successor(graph)
    assert len(graph) == 20 and graph.number_of_edges() == 45
    assert (
        tuple(graph.degree[node] for node in range(20))
        == (7, 7) + (4,) * 8 + (6, 6) + (4,) * 8
    )
    for subject in (graph, after):
        symmetry = _symmetry(subject)
        assert symmetry.source_trajectory_reduction_certified
        assert symmetry.orbits == ((0,), (1,)) + tuple(
            (i, i + 1) for i in range(2, 20, 2)
        )
        assert symmetry.multiplicities == (1, 1) + (2,) * 9
        assert all(0 < value <= 1 for value in symmetry.source.capacity)
        assert all(0 < weight <= 1 for weight in symmetry.source.conductances)
    assert dict(graph.nodes) == dict(before.nodes)
    assert dict(graph.edges) == dict(after.edges)
    assert graph.graph == after.graph
    assert all(
        graph.nodes[node]["theta"] == after.nodes[node]["theta"] for node in graph
    )
    assert all(graph.nodes[node] == after.nodes[node] for node in range(1, 20))
    assert graph.nodes[0]["glyph_history"] == after.nodes[0]["glyph_history"]
    assert graph.nodes[0]["_grammar_u2_debt"] == after.nodes[0]["_grammar_u2_debt"] == 1


@pytest.mark.parametrize(
    "broken", ("form", "phase", "capacity", "weight", "nonport_target")
)
def test_independent_fine_changes_or_nonport_target_do_not_inherit_twin_claim(broken):
    graph = _graph()
    if broken == "nonport_target":
        graph = _detached_successor(graph, target=2)
    elif broken == "weight":
        graph.edges[0, 10]["weight"] = math.nextafter(graph.edges[0, 10]["weight"], 0.0)
    else:
        key = {"form": "EPI", "phase": "theta", "capacity": "nu_f"}[broken]
        graph.nodes[2][key] = math.nextafter(graph.nodes[2][key], math.inf)
    symmetry = _symmetry(graph)
    assert not symmetry.source_trajectory_reduction_certified
    if broken in ("form", "phase", "nonport_target"):
        assert not symmetry.source_membership_certified
    if broken in ("capacity", "nonport_target"):
        assert not symmetry.capacity_preserved
    if broken == "weight":
        assert symmetry.support_automorphism and not symmetry.conductance_preserved


def _phase_box(node):
    center = Q(5 * (node % 10 // 2), 4)
    return I(center - Q(1, 32), center + Q(1, 32))


def _reference_turn(left, right):
    difference = right % 10 // 2 - left % 10 // 2
    return 1 if difference == 4 else -1 if difference == -4 else 0


def test_all_45_closed_phase_boxes_have_positive_frozen_sector_margins():
    graph, pi = _graph(), pi_interval()
    margins = []
    for left, right in graph.edges:
        gap = (
            _phase_box(right) - _phase_box(left) - 2 * _reference_turn(left, right) * pi
        )
        margin = pi.lo / 2 - gap.abs_max
        assert margin > MARGIN
        margins.append(margin)
    assert len(margins) == 45
    # Each absolute chart still belongs to the earlier 1/8 source neighborhood.
    for node in range(20):
        reference = Q(2 * (node % 10 // 2), 5) * pi
        assert (_phase_box(node) - reference).abs_max < Q(1, 8)


def test_reference_integers_fix_both_original_oriented_cycle_windings():
    pi = pi_interval()
    for offset in (0, 10):
        cycle = tuple(offset + 2 * index for index in range(5))
        edges = tuple(zip(cycle, cycle[1:] + cycle[:1]))
        turns = tuple(_reference_turn(left, right) for left, right in edges)
        assert sum(turns) == -1
        assert (
            sum(
                (
                    Q(5 * (right % 10 // 2), 4) - Q(5 * (left % 10 // 2), 4)
                    for left, right in edges
                ),
                Q(0),
            )
            == 0
        )
        for (left, right), turns in zip(edges, turns):
            gap = _phase_box(right) - _phase_box(left) - 2 * turns * pi
            assert 0 < gap.lo <= gap.hi < pi.lo / 2
        # Shared node phases telescope exactly on the cycle; only the retained
        # integer shifts remain. Independent marginal interval sums would lose
        # this correlation and are not used to infer the winding integer.


@pytest.mark.parametrize("variant", range(3))
def test_complete_shared_law_respects_independent_degree_and_weight_rate_bounds(
    variant,
):
    graph = _detached_successor(_graph(variant))
    source, pi = _field(graph), pi_interval()
    numerators = source.phase_rate_numerators()
    diameter = max(source.epi) - min(source.epi)
    assert diameter < DIAMETER
    for node, capacity, degree, form_rate, phase_rate in zip(
        source.nodes,
        source.capacity,
        source.degrees,
        source.form_rates,
        source.phase_rates,
        strict=True,
    ):
        # This exact weighted row sum retains the genuine degree: conductance
        # strength is not substituted for the bare phase-neighbor count.
        weighted_count = sum((Q(data["weight"]) for data in graph[node].values()), Q(0))
        assert weighted_count <= degree
        assert form_rate.abs_max < capacity / pi.lo
        assert abs(numerators[node]) <= capacity * weighted_count * diameter / degree
        # The report's interval arithmetic has its own outward representation.
        # The exact numerator verifies the sharp real bound; pi>3 supplies
        # a strictly wider rational bound for the stored interval endpoints.
        assert phase_rate.abs_max < capacity * weighted_count * diameter / (3 * degree)
    # Stored pressure is a supplied preparation coordinate. It is deliberately
    # different from the refreshed full-law pressure used to compute these rows.
    assert any(
        not pressure.contains(Q(graph.nodes[node]["delta_nfr"]))
        for node, pressure in zip(source.nodes, source.pressure)
    )


def test_global_derivative_inequality_closes_the_predeclared_quarter_time():
    pi = pi_interval()
    # |x_i'| <= V/pi gives D(t) <= D(0)+2 V t/pi. Then each
    # |theta_i'| <= V W D(t)/pi; an edge difference has two endpoint rows.
    displacement = 2 * (DIAMETER * HORIZON + HORIZON**2 / pi.lo) / pi.lo
    rational_ceiling = 2 * (DIAMETER * HORIZON + HORIZON**2 / 3) / 3
    assert rational_ceiling == Q(23, 576)
    assert displacement < rational_ceiling < MARGIN
    assert MARGIN - rational_ceiling == Q(29, 2880) > 0


def test_common_strict_stage_preflight_preserves_complete_snapshot_without_event():
    graph = _graph()
    operator = Contraction()
    context = ValidatedSequence(
        (operator, Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    ).step(0)
    # The grid is a constructed control. The worst-corner monotonic policy
    # argument and unchanged discrete context establish the family premise.
    for form, capacity, pressure in product(FORM, PORT_CAPACITY, PRESSURE):
        graph.nodes[0].update(EPI=form, nu_f=capacity, delta_nfr=pressure)
        before = deepcopy(graph)
        validate_contraction(graph, 0)
        result = network_stage._preflight_two_phase_stage(
            graph,
            operator,
            (0,),
            sequence_context=context,
            execution_kwargs={},
        )
        assert result.exact_stage
        assert result.snapshot is not graph
        assert dict(result.snapshot.nodes) == dict(graph.nodes)
        assert dict(graph.nodes) == dict(before.nodes)
        assert dict(graph.edges) == dict(before.edges)
        assert graph.graph == before.graph
        assert not hasattr(graph, "_last_operator_applied")
    assert Q(1, 2) / Q(19, 64) == Q(32, 19) < 10
