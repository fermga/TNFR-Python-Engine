"""Shared static preparation of the canonical two weighted-star contacts.

This factory executes the admitted AL/UM/SHA word only. It installs no
continuous trajectory, autonomous selector, or collective identity.
"""

from datetime import datetime, timezone
from fractions import Fraction as Q
from math import pi
from unittest.mock import patch

import networkx as nx

from ..dynamics.relational import RelationalExchangeModel
from ..operators import network_stage
from ..operators.definitions import Coupling, Emission, Silence
from ..operators.grammar_execution import ValidatedSequence
from ..operators.grammar_patterns import validate_sequence
from ..operators.word_execution import execute_network_operator_stage
from ..sdk._state import copy_graph_state

STARS = (tuple(range(5)), tuple(range(5, 10)))
COLLECTIVE_BLOCKS = ((0, 1, 2, 3), (4,), (5,), (6, 7, 8), (9,))
STAR_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
OLD_EDGES = STAR_EDGES + tuple((i + 5, j + 5) for i, j in STAR_EDGES)
OLD_WEIGHTS = (3, 3, 3, 2, 2, 2, 2) * 2
NEW_EDGES = tuple((4, other) for other in range(5, 10))
CAPACITIES = (4, 2, 2, 2, 8) * 2
POST_DEGREES = (4, 2, 2, 2, 9, 5, 3, 3, 3, 5)
PROBES = ((Q(1, 4), Q(201, 512)), (Q(15, 64), Q(3, 8)), (Q(17, 64), Q(409, 1024)))
BOOST = Q(1, 1024)
AMPLITUDE = Q(1, 16)
SCORE_TOLERANCE = Q(1, 2**46)
NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
CORE = {"compatibility_profile": "core"}


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def contact_source(mean, phase):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for edge, weight in zip(OLD_EDGES, OLD_WEIGHTS):
        graph.add_edge(*edge, weight=float(weight), length=1.0)
    for index, capacity in enumerate(CAPACITIES):
        graph.nodes[index].update(
            EPI=float(mean + (AMPLITUDE if index % 5 < 4 else 0)),
            theta=0.0 if index < 5 else float(phase),
            nu_f=float(capacity),
            delta_nfr=0.0,
            Si=0.5,
            glyph_history=[],
        )
    graph.graph.update(
        _t=7,
        RANDOM_SEED=17,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_LATENT_MAX=1,
        VF_BASAL_THRESHOLD=1 / 8,
        SHA_MIN_VF=1 / 8,
        UM_MIN_EPI=1 / 8,
        UM_MIN_VF=1 / 8,
        DELTA_PHI_MAX=pi / 2,
        UM_MAX_PHASE_DIFF=pi / 2,
        MIN_NETWORK_DEGREE_COUPLING=1,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=3 / 4,
        UM_CANDIDATE_COUNT=0,
        UM_CANDIDATE_MODE="proximity",
        EPI_MIN=-1,
        EPI_MAX=1,
        CLIP_MODE="hard",
        GLYPH_HYSTERESIS_WINDOW=8,
        GLYPH_FACTORS={
            "AL_boost": float(BOOST),
            "UM_theta_push": float(BOOST),
            "UM_vf_sync": 1 / 8,
            "UM_dnfr_reduction": 1 / 8,
            "SHA_vf_factor": 1 / 2,
        },
    )
    return graph


def contact_word():
    operators = (Emission(), Coupling(), Silence())
    assert tuple(operator.name for operator in operators) == (
        "emission",
        "coupling",
        "silence",
    )
    admission = validate_sequence(
        [operator.name for operator in operators], context=CORE
    )
    assert admission.passed
    assert admission.metadata["compatibility_profile"] == "core"
    return operators, ValidatedSequence(operators, context=CORE)


def execute_contact_word(graph):
    """Use the same word path and supplied clock for either support policy."""
    states, stages = [copy_graph_state(graph)], []
    operators, sequence = contact_word()
    with patch.object(network_stage, "datetime", _SuppliedClock):
        for index, operator in enumerate(operators):
            stages.append(
                execute_network_operator_stage(
                    graph, operator, (4,), sequence_context=sequence.step(index)
                )
            )
            states.append(copy_graph_state(graph))
    return tuple(states), tuple(stages), graph
