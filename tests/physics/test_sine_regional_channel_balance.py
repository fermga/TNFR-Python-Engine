"""Independent form/phase channel balances on exact and interval full states."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tests.physics.test_sine_regional_storage_balance import _source
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import (
    _sine_regional_channel_balance,
    bound_relational_sine_exchange,
)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(interval, value):
    assert interval.contains(Q(str(value))), (interval, value)


def _oracle(*, x, theta, capacity, edges, model, region):
    x, theta, capacity = (tuple(map(_mp, row)) for row in (x, theta, capacity))
    e, w = map(_mp, model.effective_weights)
    beta = _mp(model.storage_scale)
    n = len(x)
    neighbors = [set() for _ in x]
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    members = set(region)
    mobility = tuple(capacity[i] / len(row) for i, row in enumerate(neighbors))
    q = tuple(sum(x[i] - x[j] for j in row) for i, row in enumerate(neighbors))
    s = tuple(
        sum(mp.sin(theta[j] - theta[i]) for j in row) for i, row in enumerate(neighbors)
    )
    qr = tuple(sum(x[i] - x[j] for j in neighbors[i] if j in members) for i in region)
    sr = tuple(
        sum(mp.sin(theta[j] - theta[i]) for j in neighbors[i] if j in members)
        for i in region
    )
    f = tuple(
        sum(x[j] - x[i] for j in neighbors[i] if j not in members) for i in region
    )
    ext_s = tuple(
        sum(mp.sin(theta[j] - theta[i]) for j in neighbors[i] if j not in members)
        for i in region
    )
    fx = tuple(mobility[i] * (-e * q[i] + w * s[i] / mp.pi) for i in range(n))
    ft = tuple(w * mobility[i] * q[i] / (beta * mp.pi) for i in range(n))
    j = sum(w / mp.pi * qr[k] * mobility[i] * sr[k] for k, i in enumerate(region))
    bf = sum(w / mp.pi * qr[k] * mobility[i] * ext_s[k] for k, i in enumerate(region))
    bv = sum(w / mp.pi * sr[k] * mobility[i] * f[k] for k, i in enumerate(region))
    loss = sum(e * qr[k] * mobility[i] * q[i] for k, i in enumerate(region))
    form_rate = sum(
        (x[i] - x[j]) * (fx[i] - fx[j])
        for i, j in edges
        if i in members and j in members
    )
    phase_rate = sum(
        beta * mp.sin(theta[j] - theta[i]) * (ft[j] - ft[i])
        for i, j in edges
        if i in members and j in members
    )
    return {
        "internal_form_gradient": qr,
        "internal_sine_current": sr,
        "external_form_contrast": f,
        "external_sine_current": ext_s,
        "form_rates": tuple(fx[i] for i in region),
        "phase_rates": tuple(ft[i] for i in region),
        "internal_conversion": j,
        "boundary_form_input": bf,
        "boundary_phase_input": bv,
        "signed_form_loss": loss,
        "form_storage_rate": form_rate,
        "weighted_phase_storage_rate": phase_rate,
        "total_storage_rate": form_rate + phase_rate,
    }


@pytest.mark.parametrize("zero_loss", [False, True])
def test_general_full_rows_match_independent_channel_ledger(zero_loss):
    source = _source(zero_loss=zero_loss)
    report = source.regional_storage_balance(region=(2, 0))
    with mp.workdps(90):
        for channel in (report.regional_channels, report.complement_channels):
            expected = _oracle(
                x=source.epi,
                theta=source.phase,
                capacity=source.capacity,
                edges=source.edges,
                model=source.reference_model,
                region=channel.region_indices,
            )
            for name, value in expected.items():
                if isinstance(value, tuple):
                    for bound, point in zip(getattr(channel, name), value):
                        _inside(bound, point)
                else:
                    _inside(getattr(channel, name), value)
            assert abs(
                expected["form_storage_rate"]
                - expected["internal_conversion"]
                - expected["boundary_form_input"]
                + expected["signed_form_loss"]
            ) < mp.mpf("1e-85")
            assert abs(
                expected["weighted_phase_storage_rate"]
                + expected["internal_conversion"]
                - expected["boundary_phase_input"]
            ) < mp.mpf("1e-85")
            for name in (
                "form_balance_residual",
                "phase_balance_residual",
                "total_balance_residual",
            ):
                assert getattr(channel, name).contains(0)
                assert getattr(channel, name).abs_max < Q(1, 10**12)
    assert (
        report.regional_channels.total_storage_rate
        - report.direct_regional_storage_rate
    ).contains(0)
    assert (
        report.complement_channels.total_storage_rate
        - report.direct_complement_storage_rate
    ).contains(0)
    # Capacityzero at node0 is retained in the region's requested order.
    assert report.regional_channels.form_rates[1] == I(0)
    assert report.regional_channels.phase_rates[1] == I(0)
    assert report.regional_channels.flat_phase_storage_acceleration is None


def test_signed_regional_loss_is_not_global_dissipation_or_conjugate_work():
    graph = nx.path_graph(3)
    for i, x in enumerate((0, 1, 6)):
        graph.nodes[i].update(EPI=x, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    report = bound_relational_sine_exchange(
        graph, reference_model=RelationalExchangeModel(1, phase_domain="regular")
    ).regional_storage_balance(region=(0, 1))
    channels = report.regional_channels
    assert channels.signed_form_loss == I(Q(-1, 2))
    assert channels.form_storage_rate == I(Q(1, 2))
    assert channels.internal_conversion == I(0)
    assert channels.boundary_form_input == channels.boundary_phase_input == I(0)
    assert report.regional_loss == Q(9, 2)
    assert report.regional_boundary_form_work == I(5)
    assert report.comparison.continuous_loss == 17


def test_whole_support_and_empty_complement_preserve_global_conversion():
    source = _source()
    report = source.regional_storage_balance(region=source.nodes)
    regional, empty = report.regional_channels, report.complement_channels
    assert empty.region_indices == ()
    assert empty.internal_conversion == empty.total_storage_rate == I(0)
    assert empty.flat_phase_storage_acceleration == I(0)
    assert regional.boundary_form_input == regional.boundary_phase_input == I(0)
    assert (regional.signed_form_loss - source.continuous_loss).contains(0)
    assert (
        regional.weighted_phase_storage_rate + regional.internal_conversion
    ).contains(0)
    assert (regional.total_storage_rate + source.continuous_loss).contains(0)


def test_interval_kernel_retains_the_full_box_and_both_boundary_channels():
    model = RelationalExchangeModel(
        Q(3, 2), epi_weight=Q(1, 4), phase_weight=Q(3, 4), phase_domain="regular"
    )
    forms = (I(Q(-1, 8), Q(1, 8)), I(Q(7, 8), Q(9, 8)), I(Q(47, 8), Q(49, 8)))
    phases = (I(Q(1, 8), Q(3, 8)), I(Q(-3, 8), Q(-1, 8)), I(Q(3, 8), Q(5, 8)))
    capacity = (0, 1, 2)
    channel = _sine_regional_channel_balance(
        reference_model=model,
        edges=((0, 1), (1, 2)),
        degrees=(1, 2, 1),
        capacity=capacity,
        epi=forms,
        phase=phases,
        region_indices=(0, 1),
    )
    with mp.workdps(90):
        for attrs in (("lo", "hi", "lo"), ("hi", "lo", "hi"), ("midpoint",) * 3):
            x = tuple(getattr(value, attr) for value, attr in zip(forms, attrs))
            theta = tuple(
                getattr(value, attr) for value, attr in zip(phases, reversed(attrs))
            )
            expected = _oracle(
                x=x,
                theta=theta,
                capacity=capacity,
                edges=((0, 1), (1, 2)),
                model=model,
                region=(0, 1),
            )
            for name, value in expected.items():
                if isinstance(value, tuple):
                    for bound, point in zip(getattr(channel, name), value):
                        _inside(bound, point)
                else:
                    _inside(getattr(channel, name), value)
    assert channel.form_storage_rate.width > 0
    assert channel.weighted_phase_storage_rate.width > 0
    assert channel.form_balance_residual.contains(0)
    assert channel.phase_balance_residual.contains(0)
    assert channel.flat_phase_storage_acceleration is None


def test_equal_storage_can_have_different_flat_phase_curvature():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    reports = []
    for donor_form in (-3, 3):
        for i in graph:
            graph.nodes[i].update(
                EPI=3 if i == 10 else donor_form if i == 1 else 0,
                theta=0,
                nu_f=1,
            )
        reports.append(
            bound_relational_sine_exchange(
                graph, reference_model=model
            ).regional_storage_balance(region=(5, 6, 7, 8, 9))
        )
    opposite, same = reports
    assert opposite.regional_storage == same.regional_storage == I(0)
    assert opposite.complement_storage == same.complement_storage == I(Q(27, 2))
    assert opposite.boundary_storage == same.boundary_storage == I(9)
    with mp.workdps(90):
        _inside(
            opposite.regional_channels.flat_phase_storage_acceleration, 6 / mp.pi**2
        )
        _inside(same.regional_channels.flat_phase_storage_acceleration, 2 / mp.pi**2)
    assert (
        opposite.regional_channels.flat_phase_storage_acceleration
        - 3 * same.regional_channels.flat_phase_storage_acceleration
    ).contains(0)


def test_flat_phase_curvature_uses_beta_and_full_support_phase_rates():
    graph = nx.path_graph(3)
    for i, x in enumerate((0, 1, 6)):
        graph.nodes[i].update(EPI=x, theta=Q(1, 4) if i != 2 else 1, nu_f=i)
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(
        Q(3, 2), epi_weight=Q(1, 4), phase_weight=Q(3, 4), phase_domain="regular"
    )
    report = bound_relational_sine_exchange(graph, reference_model=model)
    channels = report.regional_storage_balance(region=(0, 1)).regional_channels
    # The internal phase gradient vanishes despite a nonflat exterior. The
    # full phase rates are (0,-1/pi,...) so (beta V_R)''=3/(2pi^2).
    with mp.workdps(90):
        _inside(channels.flat_phase_storage_acceleration, mp.mpf(3) / (2 * mp.pi**2))
    assert channels.weighted_phase_storage_rate == I(0)


def test_poisoned_cached_rows_do_not_reach_the_new_nested_evidence():
    source = _source()
    poisoned = replace(
        source,
        form_gradient=(999,) * 5,
        relative_resultant=((I(999), I(999)),) * 5,
        form_rates=(I(999),) * 5,
        phase_rates=(I(999),) * 5,
        storage=I(999),
    )
    expected = source.regional_storage_balance(region=(2, 0))
    actual = poisoned.regional_storage_balance(region=(2, 0))
    assert actual.regional_channels == expected.regional_channels
    assert actual.complement_channels == expected.complement_channels


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True, 0, 1)},
        {"phase": (0, float("inf"), 0)},
        {"capacity": (1, -1, 1)},
        {"degrees": (True, 2, 1)},
        {"region_indices": (0, 0)},
    ],
)
def test_interval_kernel_rejects_invalid_consumed_scalar_rows(changes):
    arguments = dict(
        reference_model=RelationalExchangeModel(1, phase_domain="regular"),
        edges=((0, 1), (1, 2)),
        degrees=(1, 2, 1),
        capacity=(1, 1, 1),
        epi=(0, 1, 6),
        phase=(0, 0, 0),
        region_indices=(0, 1),
    )
    with pytest.raises((TypeError, ValueError)):
        _sine_regional_channel_balance(**(arguments | changes))


def test_additive_projection_preserves_channel_and_legacy_availability():
    report = _source().regional_storage_balance(region=(2, 0))
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-regional-storage-balance.v1"
    assert payload["report"]["regional_channels"]["clock"] == "structural_t"
    assert payload["report"]["regional_channels"]["region_indices"] == [2, 0]
    legacy = replace(report, regional_channels=None, complement_channels=None).to_dict()
    assert legacy["report"]["regional_channels"] is None
    assert legacy["report"]["complement_channels"] is None
