"""Independent algebra and observer checks; no maintained-source forecast."""

import importlib.util
from collections import deque
from dataclasses import replace
from fractions import Fraction as Q
from math import pi
from pathlib import Path

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import cos, pi_interval
from tnfr.mathematics._validated_metric import validated_metric_taylor_step
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research.collective_form_capacity_maintenance import (
    ORDER,
    STEP,
    _observation_bins,
    admit_contact,
    identity_squared,
)


@pytest.fixture(scope="module")
def source():
    """A synthetic symmetric contact with deliberately distinct target weights."""
    graph = nx.Graph()
    for node in range(20):
        graph.add_node(
            node,
            EPI=0.25 + (0.001 if node == 0 else 0),
            theta=2 * pi * (node % 10 // 2) / 5,
            nu_f=0.5 if node < 2 else 1.0,
            delta_nfr=0.0,
        )
    for i in range(20):
        for j in range(i + 1, 20):
            if i // 10 == j // 10 and ((i % 10 // 2 - j % 10 // 2) % 5) in (1, 4):
                graph.add_edge(i, j, weight=1.0)
    for i, j, weight in (
        (0, 1, 0.875),
        (0, 10, 0.9375),
        (0, 11, 0.9375),
        (1, 10, 0.8125),
        (1, 11, 0.8125),
    ):
        graph.add_edge(i, j, weight=weight)
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )


def test_contact_metric_is_positive_and_reference_generator_is_skew(source):
    reduction, _, metric, _ = admit_contact(source)
    assert exact_symmetric_semidefinite(metric, strict=True)
    size = len(reduction.orbits)
    rho = tuple(
        Q(m * d) / nu
        for m, d, nu in zip(
            reduction.multiplicities,
            reduction.reduced_degrees,
            reduction.reduced_capacity,
            strict=True,
        )
    )
    total = sum(rho)
    form = tuple(
        tuple(metric[i][j] - rho[i] * rho[j] / total for j in range(size))
        for i in range(size)
    )
    phase = tuple(
        tuple(metric[size + i][size + j] - rho[i] * rho[j] / total for j in range(size))
        for i in range(size)
    )
    # Omit the positive common 1/pi from both rows; exact cancellation remains.
    generator = tuple(
        tuple(
            (
                (-phase[i][j] / rho[i] if b == 0 else form[i][j] / rho[i])
                if bb != b
                else Q(0)
            )
            for bb in range(2)
            for j in range(size)
        )
        for b in range(2)
        for i in range(size)
    )
    weighted = tuple(
        tuple(
            sum((metric[i][k] * generator[k][j] for k in range(2 * size)), Q(0))
            for j in range(2 * size)
        )
        for i in range(2 * size)
    )
    assert all(
        weighted[i][j] + weighted[j][i] == 0
        for i in range(2 * size)
        for j in range(2 * size)
    )
    assert form[0][6] != form[1][6]  # The asymmetric target boundary was retained.


def test_growth_constants_and_internal_period_are_proved_without_a_trajectory():
    pi_bound = pi_interval()
    cosine = cos(2 * pi_bound / 5)
    assert Q(309, 1000) < cosine.lo < cosine.hi < Q(31, 100)
    assert (Q(5, 28) + Q(1, 1000)) / Q(309, 1000) < Q(3, 5)
    assert Q(3, 5) / pi_bound.lo < Q(1, 5)
    assert 4 * Q(22, 7) ** 2 < 40


def test_identity_matches_fine_reconstruction_and_keeps_relative_origins(source):
    reduction, _, _, references = admit_contact(source)
    forms = reduction.source_form_coordinates
    phases = reduction.source_phase_coordinates
    observed = identity_squared(forms + phases, reduction.multiplicities, references)
    expanded = tuple(forms[cell] for cell in reduction.node_orbits) + tuple(
        phases[cell] for cell in reduction.node_orbits
    )
    full_refs = tuple(references[cell] for cell in reduction.node_orbits)
    full = identity_squared(expanded, (1,) * 20, full_refs)
    assert max(observed.lo, full.lo) <= min(observed.hi, full.hi)
    shifted = tuple(
        value + (Q(1, 10) if node < 10 else 0)
        for node, value in zip(reduction.representative_indices, forms, strict=True)
    )
    assert identity_squared(
        shifted + phases, reduction.multiplicities, references
    ).lo > Q(1, 100)


def test_contact_rebuild_rejects_capacity_or_support_outside_growth_premises(source):
    with pytest.raises(ValueError, match="declared"):
        admit_contact(
            replace(
                source,
                reference_model=RelationalExchangeModel(
                    Q(1, 2), epi_weight=0, phase_domain="regular"
                ),
            )
        )
    with pytest.raises(ValueError, match="capacities"):
        admit_contact(replace(source, capacity=(Q(2),) + source.capacity[1:]))
    with pytest.raises(ValueError):
        admit_contact(replace(source, conductances=(Q(2),) + source.conductances[1:]))


def test_substep_observer_encloses_exact_affine_motion_without_restarting():
    def flow(values):
        zero = values[0] * 0
        return (zero + 1, zero - 1, zero, zero)

    metric = tuple(tuple(Q(i == j) for j in range(4)) for i in range(4))
    step, _, reason = validated_metric_taylor_step(
        (Q(0),) * 4,
        Q(0),
        metric,
        STEP,
        flow,
        lambda values: (Q(1),),
        growth_rate=Q(0),
        order=ORDER,
    )
    assert reason is None
    bins = _observation_bins(step, flow, (1, 1), (Q(0), Q(0)))
    for index, bound in enumerate(bins):
        lower, upper = STEP * index / 8, STEP * (index + 1) / 8
        assert bound.lo <= 2 * lower**2
        assert bound.hi >= 2 * upper**2
    assert step.endpoint_radius < Q(1, 10**30)


def test_evaluator_captures_canonical_u6_without_executing_an_event():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/collective_form_capacity/maintenance_v1/evaluate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "synthetic_maintenance_instruments", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    for node in graph:
        graph.nodes[node].update(EPI=0.25, theta=0.0, nu_f=1.0, delta_nfr=0.0)
    before = module.capture_graph(graph)
    graph.nodes[0]["delta_nfr"] = 0.125
    after = module.capture_graph(graph)
    report = module.u6_observation(graph, before, after, "synthetic_pressure_only")
    assert report.confined
    assert report.mean_absolute_drift == 0.0625
    assert report.reference == "synthetic_pressure_only"
    assert before["fields"].epi == after["fields"].epi
    assert before["fields"].phase == after["fields"].phase


def test_synthetic_complete_word_captures_and_serializes_every_live_stage(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/collective_form_capacity/maintenance_v1/evaluate.py"
    )
    spec = importlib.util.spec_from_file_location("synthetic_maintenance_word", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 1.0, "weight")
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"SHA_vf_factor": 0.5},
        SHA_MIN_VF=0.125,
        EPI_LATENT_MAX=1.0,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        _t=7,
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25,
            theta=(0.0, 0.007, 0.02)[node],
            nu_f=0.5,
            delta_nfr=0.001,
            glyph_history=deque(("AL", "UM", "SHA", "IL", "OZ"), maxlen=8),
        )

    def refresh(subject):
        fields = module.bound_relational_sine_support_exchange(
            subject, reference_model=module.MODEL
        )
        for node, pressure in zip(fields.nodes, fields.pressure, strict=True):
            module.set_dnfr(
                subject,
                node,
                module.finite_represented_real(pressure.midpoint, "synthetic pressure")[
                    0
                ],
            )

    operators = (module.Contraction(), module.Coherence(), module.Silence())
    sequence = module.ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    captures = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module.network_stage, "datetime", module.SuppliedClock)
        refresh(graph)
        for index, operator in enumerate(operators):
            before = module.capture_graph(graph)
            assert all(
                module.validate_candidate(graph, node, operator.glyph.value).allowed
                for node in (0, 1)
            )
            stage = module.execute_network_operator_stage(
                graph,
                operator,
                (0, 1),
                sequence_context=sequence.step(index),
                compute_delta_nfr=refresh,
                include_epi_jump_certificate=False,
            )
            after = module.capture_graph(graph)
            captures.append(
                {
                    "before": before,
                    "stage": stage,
                    "after": after,
                    "u6": module.u6_observation(
                        graph, before, after, operator.glyph.value
                    ),
                }
            )
    assert all(module.node_debt(graph.nodes[node]) == 0 for node in (0, 1))
    target = tmp_path / "synthetic-complete-word.json"
    module.write_json_once(target, module.encoded(captures))
    restored = module.read(target)
    assert tuple(stage["stage"]["glyph"] for stage in restored) == ("NUL", "IL", "SHA")
    recovered = module.restore_recorded_graph_state(
        restored[-1]["after"]["state"]
    ).graph
    assert tuple(recovered.nodes[0]["glyph_history"])[-3:] == ("NUL", "IL", "SHA")
