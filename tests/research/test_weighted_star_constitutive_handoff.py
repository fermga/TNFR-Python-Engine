"""Own-law static controls on retained word records; never replay a producer."""

from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import integrators
from tnfr.operators import network_stage, word_execution
from tnfr.research import weighted_star_constitutive_handoff as owner
from tnfr.research import weighted_star_contact, weighted_star_finite_episode
from tnfr.research._recorded_graph_containers import (
    decode_recorded_container,
    encode_recorded_container,
)
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
)
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.research.weighted_star_constitutive_handoff import (
    EPSILON,
    _prefix_fields,
    compare_weighted_star_constitutive_handoff,
)
from tnfr.utils.io import json_loads

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/assets/weighted_star_registration_episode"
)


@pytest.fixture(scope="module", autouse=True)
def no_event_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("the constitutive handoff must not replay events or trajectories")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (weighted_star_contact, ("contact_source", "execute_contact_word")),
            (
                weighted_star_finite_episode,
                ("prepare_source", "evaluate_protocol", "execute_contact_word"),
            ),
            (integrators, ("rk4_coupled_step",)),
            (word_execution, ("execute_network_operator_stage",)),
            (network_stage, ("execute_pointwise_stage", "execute_coupling_stage")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    raw = read_bytes_bounded(EVIDENCE / "preparation.json", max_bytes=256 * 1024)
    preparation = decode_exact_tree(json_loads(raw))
    return preparation, compare_weighted_star_constitutive_handoff(preparation)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _encloses(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _independent(mp, source, epsilon):
    """Whole-edge high precision calculation, independent of shared work rows."""
    n = len(source.nodes)
    positions = {node: i for i, node in enumerate(source.nodes)}
    current = [mp.mpf(0) for _ in range(n)]
    gradient = [mp.mpf(0) for _ in range(n)]
    form_storage, phase_storage = mp.mpf(0), mp.mpf(0)
    epsilon = _mp(mp, epsilon)
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = positions[left], positions[right]
        form = _mp(mp, source.epi[i] - source.epi[j])
        gap = _mp(mp, source.phase[j] - source.phase[i])
        conductance = _mp(mp, weight)
        sine, cosine = mp.sin(gap), mp.cos(gap)
        edge_current = sine + epsilon * sine**3
        current[i] += edge_current
        current[j] -= edge_current
        gradient[i] += conductance * form
        gradient[j] -= conductance * form
        form_storage += conductance * form**2 / 2
        if gap:
            phase_storage += (
                1 - cosine + epsilon * (mp.mpf(2) / 3 - cosine + cosine**3 / 3)
            )
    pressure = tuple(current[i] / (mp.pi * source.degrees[i]) for i in range(n))
    form_rate = tuple(_mp(mp, source.capacity[i]) * pressure[i] for i in range(n))
    phase_rate = tuple(
        _mp(mp, source.capacity[i]) * gradient[i] / (mp.pi * source.degrees[i])
        for i in range(n)
    )
    form_work = tuple(gradient[i] * form_rate[i] for i in range(n))
    phase_work = tuple(-current[i] * phase_rate[i] for i in range(n))
    return {
        "pressure": pressure,
        "form_rates": form_rate,
        "phase_rates": phase_rate,
        "form_work": form_work,
        "phase_work": phase_work,
        "phase_storage": phase_storage,
        "storage": form_storage + phase_storage,
    }


def test_retained_byte_association_and_full_context_are_preserved(retained):
    preparation, result = retained
    freeze = json_loads((EVIDENCE / "freeze.json").read_text(encoding="utf-8"))
    receipt = next(
        row for row in freeze["artifacts"] if row["path"] == "preparation.json"
    )
    assert (
        file_receipt(EVIDENCE / "preparation.json", label="preparation.json") == receipt
    )
    assert receipt["sha256"] == (
        "059163f68d8d6f640feb4b5c50f5401385addc57548ffba3b93a2fb754131cfd"
    )
    assert result.preparation == preparation
    assert result.preparation is not preparation
    assert result.preparation_record_sha256 == (
        "8c472d31ad2b99b78c3d8e1a414ad3d5c8bf2fa17c783010bf79a0ecc6209335"
    )
    assert result.graph_time == 7
    for i, prefix in enumerate(result.prefixes):
        assert len(prefix.source.edges) == (14, 14, 19, 19)[i]
        assert prefix.stored_pressure == (Q(0),) * 10
    # The full retained records include lifecycle and grammar, not just y.
    target = restore_recorded_graph_state(
        result.preparation["static_states"][-1]
    ).graph.nodes[4]
    assert tuple(target["glyph_history"]) == ("AL", "UM", "SHA")
    assert target["latent"] is True
    assert target["_structural_lineage"]["activation_count"] == 1


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("name", ("sine", "cubic"))
def test_every_prefix_has_independent_own_law_rates_storage_and_balance(
    retained, mp, index, name
):
    prefix = retained[1].prefixes[index]
    actual = getattr(prefix, name)
    expected = _independent(mp, prefix.source, actual.epsilon)
    for field in ("pressure", "form_rates", "phase_rates", "form_work", "phase_work"):
        for bound, value in zip(getattr(actual, field), expected[field]):
            _encloses(mp, bound, value)
    for field in ("phase_storage", "storage"):
        _encloses(mp, getattr(actual, field), expected[field])
    # Independent power cancellation, followed by the computed outward bounds.
    assert abs(sum(expected["form_work"]) + sum(expected["phase_work"])) < mp.mpf(
        "1e-95"
    )
    assert actual.storage_rate.contains(0)
    # The shared resultant owner retains its own trigonometric allowance;
    # outward dyadic arithmetic does not turn that input into exact sine.
    assert actual.storage_rate.width < Q(1, 10**16)
    for rate in (actual.weighted_form_mean_rate, actual.weighted_phase_mean_rate):
        assert rate.contains(0) and rate.width < Q(1, 10**16)


def test_actual_postword_discriminator_is_not_a_common_clock_change(retained, mp):
    result = retained[1]
    prefix = result.prefixes[-1]
    alpha, c = _mp(mp, Q(201, 512)), _mp(mp, Q(63, 1024))
    factor = mp.sin(alpha) ** 3 / (16 * mp.pi)
    pressure_coefficients = (
        0,
        0,
        0,
        0,
        Q(5, 9),
        Q(-1, 5),
        Q(-1, 3),
        Q(-1, 3),
        Q(-1, 3),
        Q(-1, 5),
    )
    form_coefficients = (
        0,
        0,
        0,
        0,
        Q(65, 32),
        Q(-4, 5),
        Q(-2, 3),
        Q(-2, 3),
        Q(-2, 3),
        Q(-8, 5),
    )
    for bounds, coefficients in (
        (result.pressure_difference, pressure_coefficients),
        (result.form_rate_difference, form_coefficients),
    ):
        for bound, coefficient in zip(bounds, coefficients):
            _encloses(mp, bound, factor * _mp(mp, coefficient))
    assert result.form_rate_difference[4].lo > 0
    assert prefix.sine.phase_rates == prefix.cubic.phase_rates
    assert prefix.phase_rate_numerators[0] == Q(63, 512)
    _encloses(mp, prefix.cubic.phase_rates[0], 2 * c / mp.pi)
    assert prefix.cubic.phase_rates[0].lo > 0
    assert prefix.cubic.storage.hi < Q(1, 2)
    assert prefix.cubic.storage.lo > prefix.sine.storage.hi
    assert prefix.cubic.storage.hi < Q(603, 1280)


def test_jumps_use_each_laws_own_storage_and_keep_raw_pressure(retained, mp):
    result = retained[1]
    alpha = _mp(mp, Q(201, 512))
    cosine = mp.cos(alpha)
    extra = 5 * (mp.mpf(2) / 3 - cosine + cosine**3 / 3) / 16
    for works in (result.sine_event_work, result.cubic_event_work):
        _encloses(mp, works[0], _mp(mp, Q(-127, 262144)))
        _encloses(mp, works[2], mp.mpf(0))
    _encloses(mp, result.cubic_event_work[1] - result.sine_event_work[1], extra)
    assert result.cubic_event_work[1].lo > result.sine_event_work[1].hi
    for prefix in result.prefixes[2:]:
        assert prefix.stored_pressure[4] == 0
        assert prefix.sine.pressure[4].lo > 0
        assert prefix.cubic.pressure[4].lo > prefix.sine.pressure[4].hi


def test_cubic_potential_derivative_and_nonunit_zero_weight_incidence(mp):
    graph = nx.Graph()
    for node, (x, theta, nu) in enumerate(
        (
            (Q(-1, 4), Q(1, 8), Q(3, 2)),
            (Q(1, 2), Q(-1, 4), Q(1, 2)),
            (Q(3, 4), Q(1, 2), Q(2)),
        )
    ):
        graph.add_node(
            node, EPI=float(x), theta=float(theta), nu_f=float(nu), delta_nfr=0.0
        )
    graph.add_edge(0, 1, weight=2.5)
    graph.add_edge(1, 2, weight=0.0)
    graph.add_edge(0, 2, weight=0.25)
    prefix = _prefix_fields(graph, 0)
    expected = _independent(mp, prefix.source, EPSILON)
    assert prefix.source.degrees == (2, 2, 2)
    for field in ("pressure", "form_rates", "phase_rates"):
        for bound, value in zip(getattr(prefix.cubic, field), expected[field]):
            _encloses(mp, bound, value)
    for delta in (mp.mpf(0), mp.mpf(1) / 8, -mp.mpf(3) / 4):
        potential = (
            lambda t: 1
            - mp.cos(t)
            + (mp.mpf(2) / 3 - mp.cos(t) + mp.cos(t) ** 3 / 3) / 16
        )
        derivative = mp.diff(potential, delta)
        assert abs(derivative - (mp.sin(delta) + mp.sin(delta) ** 3 / 16)) < mp.mpf(
            "1e-95"
        )


@pytest.mark.parametrize("name", ("sine", "cubic"))
def test_nontrivial_five_block_state_has_the_complete_inherited_field(
    retained, mp, name
):
    graph = restore_recorded_graph_state(retained[0]["static_states"][-1]).graph
    blocks = ((0, 1, 2, 3), (4,), (5,), (6, 7, 8), (9,))
    multiplicities = (4, 1, 1, 3, 1)
    mobility = (Q(1), Q(13, 32), Q(4, 5), Q(2, 3), Q(8, 5))
    form = (Q(-1, 4), Q(3, 16), Q(5, 32), Q(-3, 8), Q(1, 8))
    phase = (Q(-3, 16), Q(1, 8), Q(-1, 16), Q(1, 4), Q(3, 8))
    for block, members in enumerate(blocks):
        for node in members:
            graph.nodes[node]["EPI"] = float(form[block])
            graph.nodes[node]["theta"] = float(phase[block])
            if "phase" in graph.nodes[node]:
                graph.nodes[node]["phase"] = float(phase[block])
    actual = getattr(_prefix_fields(graph, 0), name)
    epsilon = _mp(mp, actual.epsilon)
    u, v = Q(graph.edges[4, 5]["weight"]), Q(graph.edges[4, 9]["weight"])
    # Independent quotient edge table: bare multiplicity differs from summed
    # conductance. Internal edges inside A have exactly zero differences.
    coarse_edges = (
        (0, 1, 4, Q(8)),
        (1, 2, 1, u),
        (1, 3, 3, 3 * u),
        (1, 4, 1, v),
        (2, 3, 3, Q(9)),
        (2, 4, 1, Q(2)),
        (3, 4, 3, Q(6)),
    )
    current, gradient = [mp.mpf(0)] * 5, [mp.mpf(0)] * 5
    form_storage, phase_storage = mp.mpf(0), mp.mpf(0)
    for i, j, count, weight in coarse_edges:
        delta = _mp(mp, phase[j] - phase[i])
        difference = _mp(mp, form[i] - form[j])
        sine, cosine = mp.sin(delta), mp.cos(delta)
        transfer = count * (sine + epsilon * sine**3)
        current[i] += transfer
        current[j] -= transfer
        gradient[i] += _mp(mp, weight) * difference
        gradient[j] -= _mp(mp, weight) * difference
        form_storage += _mp(mp, weight) * difference**2 / 2
        phase_storage += count * (
            1 - cosine + epsilon * (mp.mpf(2) / 3 - cosine + cosine**3 / 3)
        )
    assert phase[2] != phase[3] and phase[3] != phase[4]
    for block, members in enumerate(blocks):
        factor = _mp(mp, mobility[block]) / (multiplicities[block] * mp.pi)
        for node in members:
            _encloses(mp, actual.form_rates[node], factor * current[block])
            _encloses(mp, actual.phase_rates[node], factor * gradient[block])
    _encloses(mp, actual.phase_storage, phase_storage)
    _encloses(mp, actual.storage, form_storage + phase_storage)


def _replace_graph(preparation, index, change):
    changed = deepcopy(preparation)
    original = decode_recorded_container(changed["static_states"][index])
    graph = restore_recorded_graph_state(changed["static_states"][index]).graph
    change(graph)
    records = list(changed["static_states"])
    updated = decode_recorded_container(record_graph_state(graph))
    # Reconstruction materializes NetworkX views. Preserve the original
    # omission inventory so only the intended primitive/context edit differs.
    for key in (
        "omitted_rebuildable_caches",
        "omitted_networkx_views",
        "omitted_networkx_cache",
    ):
        updated[key] = original[key]
    records[index] = encode_recorded_container(updated)
    changed["static_states"] = tuple(records)
    return changed


@pytest.mark.parametrize(
    "index,change",
    (
        (0, lambda g: g.graph.update(_t=7.0)),
        (1, lambda g: g.graph.update(_t=True)),
        (2, lambda g: g.graph.update(GAMMA={"type": "constant", "value": 1})),
        (0, lambda g: g.graph.update(compute_delta_nfr=None)),
        (1, lambda g: g.graph.update(UM_SYNC_VF=False)),
        (1, lambda g: g.graph.update(EPI_LATENT_MAX=True)),
        (0, lambda g: g.nodes[0].update(EPI=0.5)),
        (1, lambda g: g.nodes[4].update(delta_nfr=False, dnfr=0.0)),
        (2, lambda g: g.nodes[4].update(delta_nfr=0.125)),
        (0, lambda g: g.nodes[0].pop("delta_nfr")),
        (1, lambda g: g.nodes[4].update(nu_f=True)),
        (1, lambda g: g.edges[0, 1].update(weight=False)),
        (2, lambda g: g.nodes[4].update(glyph_history=deque(("UM", "AL"), maxlen=8))),
        (1, lambda g: g.nodes[4].update(_operator_step=True)),
        (2, lambda g: g.nodes[4].update(_grammar_u2_debt=False)),
        (2, lambda g: g.nodes[4].update(source_glyph="AL")),
    ),
)
def test_invalid_or_mismatched_raw_prefix_is_rejected(retained, index, change):
    changed = _replace_graph(retained[0], index, change)
    with pytest.raises((ValueError, TypeError)):
        compare_weighted_star_constitutive_handoff(changed)


def test_cached_primitive_claims_cannot_replace_the_recorded_source(retained):
    changed = deepcopy(retained[0])
    changed["primitive"]["form"] = (Q(0),) * 10
    with pytest.raises(ValueError, match="primitive source"):
        compare_weighted_star_constitutive_handoff(changed)


def test_unchanged_record_reconstruction_preserves_the_full_association(retained):
    preparation = retained[0]
    for index in range(4):
        preparation = _replace_graph(preparation, index, lambda graph: None)
    result = compare_weighted_star_constitutive_handoff(preparation)
    assert result.preparation_record_sha256 == retained[1].preparation_record_sha256


@pytest.mark.parametrize(
    "change",
    (
        lambda g: g.graph.update(EPI_LATENT_MAX=True),
        lambda g: g.graph.update(UM_SYNC_VF=1),
        lambda g: g.graph.update(UM_COMPAT_THRESHOLD=0.5),
        lambda g: g.graph.update(GRAMMAR_REJECTION_MODE="warn"),
        lambda g: g.graph["GLYPH_FACTORS"].update(AL_boost=0.5),
        lambda g: g.nodes[4].update(_operator_step=100),
        lambda g: g.nodes[4].update(_grammar_u2_debt=0),
        lambda g: g.nodes[4].update(_grammar_prior_coherence=True),
        lambda g: g.nodes[0].update(glyph_history=["AL"]),
        lambda g: g.nodes[4].update(emission_timestamp="2030-01-03T00:00:00+00:00"),
    ),
)
def test_coherent_all_prefix_context_tampering_is_not_a_frozen_association(
    retained, change
):
    preparation = retained[0]
    for index in range(4):
        preparation = _replace_graph(preparation, index, change)
    with pytest.raises(
        ValueError, match="full retained preparation record association"
    ):
        compare_weighted_star_constitutive_handoff(preparation)


def test_raw_container_types_are_part_of_the_record_association(retained):
    changed = _replace_graph(
        retained[0], 1, lambda g: g.nodes[4].update(glyph_history=["AL"])
    )
    with pytest.raises(
        ValueError, match="full retained preparation record association"
    ):
        compare_weighted_star_constitutive_handoff(changed)


def test_primitive_rejection_precedes_record_fingerprint(retained, monkeypatch):
    changed = _replace_graph(
        retained[0], 1, lambda g: g.nodes[4].update(delta_nfr=False, dnfr=0.0)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached the record-association comparison")

    monkeypatch.setattr(owner, "sha256_bytes", forbidden)
    with pytest.raises((ValueError, TypeError)) as failure:
        compare_weighted_star_constitutive_handoff(changed)
    assert "association" not in str(failure.value)
