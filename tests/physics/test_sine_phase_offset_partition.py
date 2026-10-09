"""Exact family admission, counterexamples and detached primitive rebuilding."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.phase_cycle_geometry import _sine_turn_term
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_phase_offset_partition


def _source(edges, *, order=None, label=lambda node: node, phase=0):
    graph = nx.Graph()
    if order is None:
        order = range(max(max(edge) for edge in edges) + 1)
    graph.add_nodes_from(label(node) for node in order)
    graph.add_edges_from((label(i), label(j)) for i, j in edges)
    for i in order:
        graph.nodes[label(i)].update(EPI=i - 3, theta=phase, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


CYCLE_LEAF_EDGES = tuple((i, (i + 1) % 5) for i in range(5)) + tuple(
    (i, i + 5) for i in range(5)
)
BLOCKS = (tuple(range(5)), tuple(range(5, 10)))
OFFSETS = tuple(Q(i, 5) for i in range(5)) * 2


@pytest.fixture(scope="module")
def source():
    return _source(CYCLE_LEAF_EDGES)


@pytest.fixture(scope="module")
def report(source):
    return assess_sine_phase_offset_partition(
        source, blocks=BLOCKS, phase_offset_turns=OFFSETS
    )


def test_private_leaf_family_uses_full_degrees_and_all_state_moments(report):
    assert report.invariance_certified and report.status == "certified"
    assert report.count_compatibility_certified
    assert report.moment_compatibility_certified
    assert report.internal_balance_certified
    assert report.normalized_counts == ((Q(2, 3), Q(1, 3)),) * 5 + ((1, 0),) * 5
    assert report.quotient_counts == ((Q(2, 3), Q(1, 3)), (1, 0))
    assert report.quotient_cosine_expressions == (
        ((), ((Q(1, 4), Q(1, 3)),)),
        (((Q(1, 4), Q(1)),), ()),
    )
    assert report.quotient_sine_expressions == (((), ()), ((), ()))
    assert report.internal_current_expressions == ((),) * 10
    assert report.reasons == ()
    # The captured nonuniform forms are intentionally outside this family.
    assert report.source.epi[:5] == (-3, -2, -1, 0, 1)
    state = report.evaluate((1, 0), (0, Q(1, 8)))
    assert state.fine_form == (1,) * 5 + (0,) * 5
    assert state.block_phase_rates == (Q(1, 3), -1)
    assert state.full_phase_rates == (Q(1, 3),) * 5 + (-1,) * 5
    assert state.full_row_equality_certified
    assert state.phase_rate_residuals == (0,) * 10
    assert state.internal_form_storage == (0, 0)
    assert state.cross_form_storage == state.full_form_storage == Q(5, 2)
    assert state.clock == "tau=t/pi"
    assert state.block_form_rates[0].lo > 0
    assert state.block_form_rates[1].hi < 0
    with pytest.raises(FrozenInstanceError):
        report.status = "excluded"


def test_synchronized_partition_is_included_without_phase_storage():
    source = _source(((0, 2), (0, 3), (1, 2), (1, 3)))
    report = assess_sine_phase_offset_partition(
        source, blocks=((0, 1), (2, 3)), phase_offset_turns=(0,) * 4
    )
    assert report.invariance_certified
    state = report.evaluate((2, -1), (0, Q(1, 4)))
    assert state.block_phase_rates == (3, -3)
    assert state.block_form_rates == (I(1), I(-1))
    assert state.full_storage_bounds == I(22)


def test_internal_real_moments_may_differ_without_breaking_family():
    source = _source(((0, 1), (1, 2), (3, 4), (3, 5), (0, 3), (1, 3), (1, 4), (2, 5)))
    report = assess_sine_phase_offset_partition(
        source,
        blocks=((0, 1, 2), (3, 4, 5)),
        phase_offset_turns=(0, 0, Q(1, 2), 0, 0, Q(1, 2)),
    )
    assert report.invariance_certified
    assert report.normalized_counts == ((Q(1, 2), Q(1, 2)),) * 6
    assert tuple(
        report.row_cosine_bounds[i][report.node_blocks[i]] for i in range(6)
    ) == tuple(map(I, (Q(1, 2), 0, -Q(1, 2), 0, Q(1, 2), -Q(1, 2))))
    state = report.evaluate((2, -3), (Q(1, 7), Q(1, 9)))
    assert state.block_phase_rates == (Q(5, 2), -Q(5, 2))
    assert state.internal_phase_storage_bounds == (I(2), I(2))


def test_true_root_of_unity_cancellation_can_remain_unavailable():
    # Fourth roots and third roots each sum to zero. The deliberately small
    # identity basis cannot prove every rotation of the third-root sum.
    source = _source(tuple((i, j) for i in range(4) for j in range(4, 7)))
    report = assess_sine_phase_offset_partition(
        source,
        blocks=((0, 1, 2, 3), (4, 5, 6)),
        phase_offset_turns=(0, Q(1, 4), Q(1, 2), Q(3, 4), 0, Q(1, 3), Q(2, 3)),
    )
    assert report.count_compatibility_certified
    assert report.internal_balance_certified
    assert not report.invariance_certified
    assert report.status == "unavailable"
    assert any("unresolved" in row for row in report.cross_moment_statuses)
    assert not any("excluded" in row for row in report.cross_moment_statuses)
    assert report.quotient_counts is None
    assert report.quotient_cosine_expressions is None
    with pytest.raises(ValueError, match="certified"):
        report.evaluate((0, 1), (0, 0))


def test_nonzero_internal_current_excludes_a_counts_and_cross_moment_candidate():
    # The common cross phase moment vanishes by antipodal cancellation;
    # each internal two-node edge nevertheless exerts a nonzero sine torque.
    source = _source(
        tuple((i, j) for i in range(4) for j in range(4, 8))
        + ((0, 1), (2, 3), (4, 5), (6, 7))
    )
    report = assess_sine_phase_offset_partition(
        source,
        blocks=((0, 1, 2, 3), (4, 5, 6, 7)),
        phase_offset_turns=(0, Q(1, 4), Q(1, 2), Q(3, 4)) * 2,
    )
    assert report.count_compatibility_certified
    assert report.moment_compatibility_certified
    assert (
        report.normalized_counts
        == ((Q(1, 5), Q(4, 5)),) * 4 + ((Q(4, 5), Q(1, 5)),) * 4
    )
    assert not report.internal_balance_certified
    assert report.status == "excluded"
    assert report.internal_current_statuses == ("excluded",) * 8
    assert report.internal_current_bounds == (I(Q(1, 5)), I(-Q(1, 5))) * 4


def test_node_and_block_order_and_integer_turn_gauges(report):
    order = (8, 0, 5, 3, 6, 9, 1, 7, 4, 2)
    source = _source(CYCLE_LEAF_EDGES, order=order, label=lambda i: f"node:{i}")
    offsets = tuple(OFFSETS[i] + 3 * i + Q(1, 11) for i in order)
    rearranged = assess_sine_phase_offset_partition(
        source,
        blocks=(
            tuple(f"node:{i}" for i in reversed(BLOCKS[1])),
            tuple(f"node:{i}" for i in reversed(BLOCKS[0])),
        ),
        phase_offset_turns=offsets,
    )
    state = rearranged.evaluate((-1, 2), (Q(1, 8), 0))
    original = report.evaluate((2, -1), (0, Q(1, 8)))
    assert rearranged.invariance_certified
    assert state.block_phase_rates == original.block_phase_rates[::-1]
    assert state.full_phase_rates == tuple(original.full_phase_rates[i] for i in order)
    assert state.full_form_rates == tuple(original.full_form_rates[i] for i in order)
    assert state.full_storage_bounds == original.full_storage_bounds


def test_evaluation_ignores_poisoned_derived_partition_and_source_fields(report):
    poisoned_source = replace(
        report.source,
        form_rates=(I(123),),
        phase_rates=(I(999),),
        storage=I(-1),
        form_gradient=(Q(1),),
    )
    poisoned = replace(
        report,
        source=poisoned_source,
        status="excluded",
        invariance_certified=False,
        normalized_counts=(),
        node_blocks=(),
        block_indices=(),
        quotient_counts=((True,),),
        quotient_cosine_expressions=(),
        quotient_sine_expressions=(),
        row_sine_bounds=(),
    )
    state = poisoned.evaluate((1, -1), (0, Q(1, 8)))
    expected = report.evaluate((1, -1), (0, Q(1, 8)))
    assert state.partition.invariance_certified
    assert state.block_form_rates == expected.block_form_rates
    assert state.block_phase_rates == expected.block_phase_rates
    assert state.full_storage_bounds == expected.full_storage_bounds
    changed = replace(report, phase_offset_turns=(Q(1, 4),) + OFFSETS[1:])
    with pytest.raises(ValueError, match="certified"):
        changed.evaluate((1, 0), (0, 0))


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
@pytest.mark.parametrize("value", (True, float("nan"), complex(1, 2)))
def test_source_primitives_are_readmitted(source, field, value):
    bad = replace(source, **{field: (value,) + getattr(source, field)[1:]})
    with pytest.raises((TypeError, ValueError)):
        assess_sine_phase_offset_partition(
            bad, blocks=BLOCKS, phase_offset_turns=OFFSETS
        )


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_model_booleans_cannot_equal_valid_coefficients(source, field):
    model = copy(source.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    with pytest.raises((TypeError, ValueError)):
        assess_sine_phase_offset_partition(
            replace(source, reference_model=model),
            blocks=BLOCKS,
            phase_offset_turns=OFFSETS,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"capacity": (0,) + (1,) * 9},
        {"degrees": (True,) + (3,) * 4 + (1,) * 5},
        {"law": "native"},
        {"edges": CYCLE_LEAF_EDGES + ((0, 5),)},
    ),
)
def test_unsupported_source_and_support_reject(source, mutation):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_phase_offset_partition(
            replace(source, **mutation), blocks=BLOCKS, phase_offset_turns=OFFSETS
        )


@pytest.mark.parametrize(
    "offsets",
    ((0,) * 9, (0,) * 11, (False,) + OFFSETS[1:], (0.0,) * 10, set(OFFSETS), "0" * 10),
)
def test_exact_turn_input_domain(source, offsets):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_phase_offset_partition(
            source, blocks=BLOCKS, phase_offset_turns=offsets
        )


@pytest.mark.parametrize(
    "blocks",
    (
        (tuple(range(10)),),
        tuple((i,) for i in range(10)),
        (tuple(range(5)), tuple(range(4, 9))),
        (set(range(5)), tuple(range(5, 10))),
    ),
)
def test_proper_ordered_partition_domain(source, blocks):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_phase_offset_partition(
            source, blocks=blocks, phase_offset_turns=OFFSETS
        )


@pytest.mark.parametrize(
    "form,turns",
    (
        ((True, 0), (0, 0)),
        ((1,), (0, 0)),
        ((1, float("inf")), (0, 0)),
        ((1, 0), (0, True)),
        ((1, 0), (0, 0.1)),
        ((1, 0), (0,)),
    ),
)
def test_detached_evaluation_admits_collective_inputs(report, form, turns):
    with pytest.raises((TypeError, ValueError)):
        report.evaluate(form, turns)


def test_evaluation_readmits_changed_primitive_source(report):
    bad = replace(
        report, source=replace(report.source, phase=(False,) + report.source.phase[1:])
    )
    with pytest.raises((TypeError, ValueError)):
        bad.evaluate((1, 0), (0, 0))
    with pytest.raises(TypeError):
        assess_sine_phase_offset_partition(
            object(), blocks=BLOCKS, phase_offset_turns=OFFSETS
        )


@pytest.mark.parametrize(
    "turn,expected",
    (
        (Q(0), None),
        (Q(1, 2), None),
        (Q(2), None),
        (Q(7, 10), (Q(1, 5), -1)),
        (Q(3, 10), (Q(1, 5), 1)),
        (-Q(1, 5), (Q(1, 5), -1)),
    ),
)
def test_shared_single_sine_term_preserves_periodic_odd_reflection_folding(
    turn, expected
):
    assert _sine_turn_term(turn, fold_circle=True) == expected
    assert _sine_turn_term(turn) == (
        None if not turn else (abs(turn), 1 if turn > 0 else -1)
    )
