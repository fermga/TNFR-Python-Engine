"""Regional form observations bind stored state to the shared nodal rate.

These finite controls admit supplied ordered frames, not a selected partition,
phase law or measured derivative. Affine closure controls use separately
declared generators and sources, not an inferred law from stored pressure.
"""

import copy
import math
from dataclasses import FrozenInstanceError
from decimal import ROUND_DOWN, Inexact, Rounded, localcontext
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics._euler_kernel import euler_update
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.errors import FrequencyError, NetworkConfigError
from tnfr.mathematics import BEPIElement
from tnfr.physics import form_geometry, observe_source_relative_form
from tnfr.sdk.simple import Network
from tnfr.types import ensure_bepi

REGIONS = ((0, 1, 2), (3, 4, 5))


def test_full_source_relative_matrix_retains_unforced_region_and_held_rate():
    graph = _graph()
    before = copy.deepcopy(graph)
    # Only the first region has source contrast. Its constant mean is irrelevant.
    source = (3, 1, 2, 7, 7, 7)
    result = observe_source_relative_form(graph, REGIONS, held_source_rate=source)
    assert result.source_mean == (2, 7)
    assert result.source_contrast_a == (2, 0)
    assert result.source_contrast_b == (0, 0)
    assert result.source_contrast_norm_squared == 2
    assert result.contrast_reconstructible
    assert result.relative_real == ((2, 0), (0, 0))
    assert result.relative_imag_numerator == ((0, 0), (12, 0))
    assert result.relative_rate_real == ((0, 0), (-1, 0))
    assert result.relative_rate_imag_numerator == ((4, 0), (6, 0))
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph
    assert list(graph.edges(data=True)) == list(before.edges(data=True))
    # Full W*c/(c†c) recovers both contrasts; the diagonal alone misses region 1.
    for i, region in enumerate(result.form.regions):
        assert result.relative_real[i][0] == region.contrast_a
        assert result.relative_imag_numerator[i][0] / 2 == region.contrast_b
    with pytest.raises(FrozenInstanceError):
        result.held_source_rate = ()
    source_list = list(source)
    detached = observe_source_relative_form(
        graph, REGIONS, held_source_rate=source_list
    )
    source_list[0] = 100
    assert detached == result


def test_source_relative_zero_limits_and_exact_subnormal_source():
    graph = _graph()
    result = observe_source_relative_form(graph, REGIONS, held_source_rate=(3,) * 6)
    assert not result.contrast_reconstructible
    assert not any(map(any, result.relative_real))
    assert not any(map(any, result.relative_imag_numerator))
    # Source coefficients are an exact declared model, not a binary64 state rate.
    tiny = Fraction(1, 10**400)
    source = (tiny, -tiny, 0, 0, 0, 0)
    result = observe_source_relative_form(graph, REGIONS, held_source_rate=source)
    assert result.contrast_reconstructible
    assert result.relative_real[0][0] == 2 * tiny
    assert result.source_contrast_norm_squared == 2 * tiny**2
    zero_form = observe_source_relative_form(
        _graph((0,) * 6), REGIONS, held_source_rate=source
    )
    assert not any(map(any, zero_form.relative_real))
    assert zero_form.relative_rate_imag_numerator[0][0] == 4 * tiny


def test_mixed_source_frame_node_order_and_shared_euler_pushforward():
    graph = _graph()
    graph.graph.update(
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
        CLIP_MODE="hard",
    )
    source = (2, -1, 3, 0, 2, -3)
    before = observe_source_relative_form(graph, REGIONS, held_source_rate=source)
    assert before.source_contrast_a == (3, -2)
    assert before.source_contrast_b == (-5, 8)
    assert before.source_contrast_norm_squared == Fraction(64, 3)
    assert before.relative_real == ((3, -2), (-5, 8))
    assert before.relative_imag_numerator == ((10, -16), (18, -12))
    assert before.relative_rate_real == ((Fraction(-5, 3), Fraction(8, 3)), (-4, 5))
    assert before.relative_rate_imag_numerator == ((6, -4), (4, 2))
    reordered = nx.Graph()
    order = (5, 2, 0, 4, 1, 3)
    reordered.add_nodes_from((node, dict(graph.nodes[node])) for node in order)
    reordered.add_edges_from(graph.edges())
    rebound = observe_source_relative_form(
        reordered, REGIONS, held_source_rate=tuple(source[node] for node in order)
    )
    assert rebound.form.nodes == order
    assert rebound.relative_real == before.relative_real
    assert rebound.relative_imag_numerator == before.relative_imag_numerator
    assert rebound.relative_rate_real == before.relative_rate_real
    h = Fraction(1, 8)
    update_epi_via_nodal_equation(graph, dt=float(h), method="euler")
    after = observe_source_relative_form(graph, REGIONS, held_source_rate=source)
    for field, rate in (
        ("relative_real", "relative_rate_real"),
        ("relative_imag_numerator", "relative_rate_imag_numerator"),
    ):
        expected = tuple(
            tuple(x + h * dx for x, dx in zip(row, rate_row, strict=True))
            for row, rate_row in zip(
                getattr(before, field), getattr(before, rate), strict=True
            )
        )
        assert getattr(after, field) == expected


def test_source_relative_moving_source_needs_separate_chain_rule_term():
    # Freeze form motion while changing the supplied source between snapshots.
    graph = _graph()
    for node in graph:
        graph.nodes[node][ALIAS_DNFR[0]] = 0
    source = (1, -1, 0, 0, 0, 0)
    first = observe_source_relative_form(graph, REGIONS, held_source_rate=source)
    second = observe_source_relative_form(
        graph, REGIONS, held_source_rate=tuple(2 * value for value in source)
    )
    assert second.relative_real[0][0] - first.relative_real[0][0] == 2
    assert not any(map(any, first.relative_rate_real))
    assert not any(map(any, first.relative_rate_imag_numerator))
    assert (
        "held_source_pushforward_excludes_source_motion_and_runtime_events"
        in first.scope
    )


def test_source_relative_source_admission_and_sdk_delegate(monkeypatch):
    from tnfr.physics import source_relative_form

    graph = _graph()
    for invalid in ({0: 1}, {1, 2}, "source", (0,) * 5, (True,) * 6, (math.nan,) * 6):
        with pytest.raises((TypeError, ValueError)):
            observe_source_relative_form(graph, REGIONS, held_source_rate=invalid)
    sentinel = object()
    calls = []

    def tracked(graph_arg, regions_arg, *, held_source_rate):
        calls.append((graph_arg, regions_arg, held_source_rate))
        return sentinel

    monkeypatch.setattr(source_relative_form, "observe_source_relative_form", tracked)
    source = (0,) * 6
    assert (
        Network(graph).source_relative_form(REGIONS, held_source_rate=source)
        is sentinel
    )
    assert calls == [(graph, REGIONS, source)]


def _graph(form=(1, -1, 0, 1, 1, -2)):
    graph = nx.path_graph(len(form))
    capacity = (1, 2, 0, 1, 1, 1)
    pressure = (1, 0.5, 3, 0, 1, -1)
    for node, value in enumerate(form):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: value,
                ALIAS_VF[0]: capacity[node],
                ALIAS_DNFR[0]: pressure[node],
            }
        )
    return graph


def _forbid_pressure_refresh(*args, **kwargs):
    raise AssertionError("a stored-state observation must not refresh pressure")


def test_stored_nodal_projection_has_independent_geometry_and_no_graph_writes(
    monkeypatch,
):
    graph = _graph()
    graph.graph.update(
        compute_delta_nfr=_forbid_pressure_refresh,
        _cache={"sentinel": [1, 2]},
        Gamma={"unused_declared_source": _forbid_pressure_refresh},
    )
    # Primitive phase and stale derivatives are not coordinates of this read-out.
    graph.nodes[0].update(theta="unused invalid phase", dEPI=999)
    before = copy.deepcopy(graph)
    calls = []
    original = form_geometry.compute_canonical_nodal_derivative

    def tracked(capacity, pressure):
        calls.append((capacity, pressure))
        return original(capacity, pressure)

    monkeypatch.setattr(form_geometry, "compute_canonical_nodal_derivative", tracked)
    report = form_geometry.observe_regional_form(graph, REGIONS)
    assert calls == [(1, 1), (2, 0.5), (0, 3), (1, 0), (1, 1), (1, -1)]
    assert graph.graph == before.graph
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert list(graph.edges(data=True)) == list(before.edges(data=True))
    assert report.nodes == tuple(range(6))
    assert report.epi == (1, -1, 0, 1, 1, -2)
    assert report.capacity == (1, 2, 0, 1, 1, 1)
    assert report.stored_pressure == (1, Fraction(1, 2), 3, 0, 1, -1)
    assert report.nodal_rate == (1, 1, 0, 0, 1, -1)
    assert report.nodal_rate_rounding_defect == (0,) * 6
    a, b = report.regions
    assert (a.nodes, b.nodes) == REGIONS
    assert (a.mean, b.mean, a.mean_rate, b.mean_rate) == (0, 0, Fraction(2, 3), 0)
    assert (a.contrast_a, a.contrast_b, b.contrast_a, b.contrast_b) == (2, 0, 0, 6)
    assert (a.contrast_a_rate, a.contrast_b_rate) == (0, 2)
    assert (b.contrast_a_rate, b.contrast_b_rate) == (-1, 3)
    assert (a.intensity, b.intensity, a.intensity_rate, b.intensity_rate) == (
        2,
        6,
        0,
        6,
    )
    assert (a.amplitude_estimate, b.amplitude_estimate) == pytest.approx(
        (math.sqrt(2), math.sqrt(6))
    )
    assert (a.phase_estimate, b.phase_estimate) == pytest.approx((0, math.pi / 2))
    assert (a.amplitude_rate_estimate, b.amplitude_rate_estimate) == pytest.approx(
        (0, math.sqrt(1.5))
    )
    assert (a.phase_rate_estimate, b.phase_rate_estimate) == pytest.approx(
        (1 / math.sqrt(3), 1 / (2 * math.sqrt(3)))
    )
    assert report.gram_real == ((2, 0), (0, 6))
    assert report.gram_imag_numerator == ((0, -12), (12, 0))
    assert report.gram_rate_real == ((0, 1), (1, 6))
    assert report.gram_rate_imag_numerator == ((0, -6), (6, 0))
    assert (
        "unforced_nu_times_stored_pressure_prediction_not_measured_derivative"
        in report.scope
    )
    assert (
        "gamma_operator_jumps_clipping_and_other_runtime_rates_excluded" in report.scope
    )
    assert all(isinstance(value, Fraction) for value in report.epi + report.nodal_rate)
    with pytest.raises(FrozenInstanceError):
        a.mean = 17
    graph.nodes[0][ALIAS_EPI[0]] = 99
    assert report.epi[0] == 1


def test_shared_euler_endpoint_obeys_quadratic_gram_identity_with_rounding_evidence():
    graph = _graph()
    graph.graph.update(
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
        CLIP_MODE="hard",
    )
    before = form_geometry.observe_regional_form(graph, REGIONS)
    h = Fraction(1, 8)
    rounding = []
    expected_epi = []
    for node, x, rate in zip(before.nodes, before.epi, before.nodal_rate, strict=True):
        endpoint = euler_update(float(x), float(h), float(rate))
        rounding.append(Fraction(endpoint) - x - h * rate)
        expected_epi.append(Fraction(endpoint))
    # This dyadic control represents the finite Euler endpoint exactly. It is
    # not an assertion that Euler is the continuous solution of a pressure law.
    assert rounding == [0] * 6
    update_epi_via_nodal_equation(graph, dt=float(h), t=0.0, method="euler")
    after = form_geometry.observe_regional_form(graph, REGIONS)
    assert after.epi == tuple(expected_epi)
    assert tuple(row.mean for row in after.regions) == (2 * h / 3, 0)
    assert tuple((row.contrast_a, row.contrast_b) for row in after.regions) == (
        (2, 2 * h),
        (-h, 6 + 3 * h),
    )
    rate_gram_real = ((Fraction(2, 3), 1), (1, 2))
    rate_gram_imag = ((0, -2), (2, 0))
    for i in range(2):
        for j in range(2):
            assert (
                after.gram_real[i][j]
                == before.gram_real[i][j]
                + h * before.gram_rate_real[i][j]
                + h**2 * rate_gram_real[i][j]
            )
            assert (
                after.gram_imag_numerator[i][j]
                == before.gram_imag_numerator[i][j]
                + h * before.gram_rate_imag_numerator[i][j]
                + h**2 * rate_gram_imag[i][j]
            )
    assert (
        after.gram_real[0][0] > before.gram_real[0][0]
    )  # The omitted h² term matters.


def test_zero_amplitude_retains_cartesian_response_without_inventing_phase():
    graph = _graph((2, 2, 2))
    for node, pressure in enumerate((1, 0, -1)):
        graph.nodes[node].update({ALIAS_VF[0]: 1, ALIAS_DNFR[0]: pressure})
    report = form_geometry.observe_regional_form(graph, ((0, 1, 2),))
    row = report.regions[0]
    assert (row.mean, row.intensity, row.intensity_rate) == (2, 0, 0)
    assert (row.contrast_a_rate, row.contrast_b_rate) == (1, 3)
    assert row.amplitude_estimate == 0
    assert (
        row.phase_estimate
        is row.amplitude_rate_estimate
        is row.phase_rate_estimate
        is None
    )
    assert dict(row.estimate_status)["phase_rate_estimate"] == "zero_amplitude"
    assert report.gram_real == report.gram_rate_real == ((0,),)
    for node, rate in zip(report.nodes, report.nodal_rate, strict=True):
        graph.nodes[node][ALIAS_EPI[0]] = euler_update(2.0, 0.25, float(rate))
    after = form_geometry.observe_regional_form(graph, ((0, 1, 2),)).regions[0]
    assert after.intensity == Fraction(1, 8)
    assert after.phase_estimate == pytest.approx(math.pi / 3)


def test_consumed_scalar_admission_and_explicit_partition_are_required():
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = ensure_bepi(-1)
    assert form_geometry.observe_regional_form(graph, REGIONS).epi[1] == -1
    nonuniform = BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0))
    invalid = (
        (ALIAS_EPI[0], True, (TypeError, ValueError)),
        (ALIAS_EPI[0], 1 + 0j, (TypeError, ValueError)),
        (ALIAS_EPI[0], nonuniform, (TypeError, ValueError)),
        (ALIAS_EPI[0], Fraction(1, 2**2000), (TypeError, ValueError)),
        (
            ALIAS_EPI[0],
            {
                "continuous": (Fraction(1, 10**400),) * 2,
                "discrete": (0, 0),
                "grid": (0, 1),
            },
            (TypeError, ValueError),
        ),
        (
            ALIAS_EPI[0],
            {"continuous": (True, True), "discrete": (True, True), "grid": (0, 1)},
            (TypeError, ValueError),
        ),
        (ALIAS_DNFR[0], False, (TypeError, ValueError)),
        (ALIAS_VF[0], -1, FrequencyError),
    )
    for field, value, error in invalid:
        candidate = _graph()
        candidate.nodes[0][field] = value
        with pytest.raises(error):
            form_geometry.observe_regional_form(candidate, REGIONS)
    del graph.nodes[0][ALIAS_DNFR[0]]
    with pytest.raises(ValueError, match="missing consumed"):
        form_geometry.observe_regional_form(graph, REGIONS)
    for partition in (
        (),
        ((0, 1), (2, 3, 4, 5)),
        ((0, 1, 2),),
        ((0, 1, 2), (2, 4, 5)),
        ((0, 1, 2), (3, 4, 99)),
        {REGIONS[0], REGIONS[1]},
    ):
        with pytest.raises((TypeError, ValueError)):
            form_geometry.observe_regional_form(_graph(), partition)
    with pytest.raises(ValueError, match="nonempty"):
        form_geometry.observe_regional_form(nx.Graph(), ())


def test_nodal_materialization_and_polar_range_have_distinct_boundaries():
    for capacity, pressure, error in (
        (math.ulp(0.0), 0.5, ValueError),
        (1e308, 1e308, NetworkConfigError),
    ):
        graph = _graph()
        graph.nodes[0].update({ALIAS_VF[0]: capacity, ALIAS_DNFR[0]: pressure})
        with pytest.raises(error):
            form_geometry.observe_regional_form(graph, REGIONS)
    graph = _graph()
    graph.nodes[0].update({ALIAS_VF[0]: 0.1, ALIAS_DNFR[0]: 0.1})
    rounded = form_geometry.observe_regional_form(graph, REGIONS)
    assert rounded.nodal_rate[0] == Fraction(0.1 * 0.1)
    assert (
        rounded.nodal_rate_rounding_defect[0]
        == Fraction(0.1 * 0.1) - Fraction(0.1) ** 2
        != 0
    )
    for magnitude, expected_status in (
        (1e200, "available"),
        (1.5e308, "unrepresentable_overflow"),
    ):
        graph = _graph((magnitude, magnitude, -magnitude))
        for node in graph:
            graph.nodes[node][ALIAS_VF[0]] = 0
        row = form_geometry.observe_regional_form(graph, ((0, 1, 2),)).regions[0]
        assert row.intensity > Fraction.from_float(
            float.fromhex("0x1.fffffffffffffp+1023")
        )
        assert dict(row.estimate_status)["amplitude_estimate"] == expected_status
        assert (row.amplitude_estimate is None) == (expected_status != "available")
        assert row.phase_estimate == pytest.approx(math.pi / 2)
    graph = _graph((1e300, -1e300, 0))
    for node, pressure in enumerate((1e-300, 1e-300, -2e-300)):
        graph.nodes[node].update({ALIAS_VF[0]: 1, ALIAS_DNFR[0]: pressure})
    row = form_geometry.observe_regional_form(graph, ((0, 1, 2),)).regions[0]
    assert row.contrast_a * row.contrast_b_rate > 0
    assert row.phase_rate_estimate is None
    assert (
        dict(row.estimate_status)["phase_rate_estimate"] == "unrepresentable_underflow"
    )


def test_polar_estimates_do_not_inherit_or_modify_decimal_context():
    graph = _graph()
    expected = form_geometry.observe_regional_form(graph, REGIONS)
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.Emax, context.Emin = 3, -3
        context.traps[Inexact] = context.traps[Rounded] = True
        before = (
            context.prec,
            context.rounding,
            context.Emax,
            context.Emin,
            dict(context.traps),
            dict(context.flags),
        )
        assert form_geometry.observe_regional_form(graph, REGIONS) == expected
        assert (
            context.prec,
            context.rounding,
            context.Emax,
            context.Emin,
            dict(context.traps),
            dict(context.flags),
        ) == before


def test_sdk_regional_form_delegates_without_state_or_partition_reconstruction(
    monkeypatch,
):
    graph = _graph()
    network = Network(graph)
    sentinel = object()
    calls = []

    def observe(actual_graph, actual_regions):
        calls.append((actual_graph, actual_regions))
        return sentinel

    monkeypatch.setattr(form_geometry, "observe_regional_form", observe)
    assert network.regional_form(REGIONS) is sentinel
    assert len(calls) == 1
    assert calls[0][0] is graph
    assert calls[0][1] is REGIONS


def _directed_affine_form_model():
    # Every block is circulant, but the two-way interface is neither reciprocal
    # nor symmetric. Rows sum to zero; this is a genuine directed generator.
    generator = (
        (-3, 1, 0, 0, 2, 0),
        (0, -3, 1, 0, 0, 2),
        (1, 0, -3, 2, 0, 0),
        (1, 0, 0, -2, 0, 1),
        (0, 1, 0, 1, -2, 0),
        (0, 0, 1, 0, 1, -2),
    )
    return generator, (1, 1, 1, -2, -2, -2)


def test_affine_closure_recovers_directed_mean_and_complex_contrast_laws():
    generator, source = _directed_affine_form_model()
    report = form_geometry.derive_regional_affine_closure(
        tuple(range(6)), REGIONS, generator=generator, source=source
    )
    assert report.all_state_closed
    assert report.block_circulant_defect == ((0,) * 6,) * 6
    assert report.source_contrast_a == report.source_contrast_b == (0, 0)
    assert report.mean_generator == ((-2, 2), (1, -1))
    assert report.mean_source == (1, -2)
    assert report.contrast_generator_real == (
        (Fraction(-7, 2), -1),
        (1, Fraction(-5, 2)),
    )
    assert report.contrast_generator_imag_over_sqrt3 == (
        (Fraction(-1, 2), -1),
        (0, Fraction(1, 2)),
    )

    # Independently evaluate G*x+b, then use the production observer to verify
    # the reduction in its actual frame. This detects a conjugated/sign-flipped
    # complex generator even if its real decay rates happen to agree.
    graph = _graph((2, -1, 5, -2, 3, 1))
    form = tuple(graph.nodes[node][ALIAS_EPI[0]] for node in graph)
    rates = tuple(
        sum(coefficient * x for coefficient, x in zip(row, form, strict=True)) + b
        for row, b in zip(generator, source, strict=True)
    )
    assert rates == (0, 11, -16, 5, -11, 4)
    for node, rate in zip(graph, rates, strict=True):
        graph.nodes[node].update({ALIAS_VF[0]: 1, ALIAS_DNFR[0]: rate})
    observation = form_geometry.observe_regional_form(graph, REGIONS)
    means = tuple(row.mean for row in observation.regions)
    a = tuple(row.contrast_a for row in observation.regions)
    b = tuple(row.contrast_b for row in observation.regions)
    for i, region in enumerate(observation.regions):
        real = report.contrast_generator_real[i]
        imag = report.contrast_generator_imag_over_sqrt3[i]
        assert (
            region.mean_rate
            == sum(
                coefficient * mean
                for coefficient, mean in zip(
                    report.mean_generator[i], means, strict=True
                )
            )
            + report.mean_source[i]
        )
        assert region.contrast_a_rate == sum(
            r * left - s * right
            for r, s, left, right in zip(real, imag, a, b, strict=True)
        )
        assert region.contrast_b_rate == sum(
            3 * s * left + r * right
            for r, s, left, right in zip(real, imag, a, b, strict=True)
        )


def test_affine_closure_rejects_distinct_generator_and_source_obstructions():
    generator, source = _directed_affine_form_model()
    changed = [list(row) for row in generator]
    changed[1][0] = Fraction(1, 8)
    noncirculant = form_geometry.derive_regional_affine_closure(
        tuple(range(6)), REGIONS, generator=changed, source=source
    )
    assert not noncirculant.all_state_closed
    assert noncirculant.block_circulant_defect == (
        (0, 0, 0, 0, 0, 0),
        (Fraction(1, 8), 0, 0, 0, 0, 0),
        *((0,) * 6,) * 4,
    )
    assert noncirculant.source_contrast_a == noncirculant.source_contrast_b == (0, 0)
    assert noncirculant.contrast_generator_real is None
    assert noncirculant.contrast_generator_imag_over_sqrt3 is None

    # A constant-in-time source can still carry contrast: constancy in time is
    # insufficient for invariance under the omitted common contrast rotation.
    for imposed, contrast_a, contrast_b in (
        ((1, -1, 0, 0, 0, 0), (2, 0), (0, 0)),
        ((1, 1, -2, 0, 0, 0), (0, 0), (6, 0)),
    ):
        directional = form_geometry.derive_regional_affine_closure(
            tuple(range(6)), REGIONS, generator=generator, source=imposed
        )
        assert not directional.all_state_closed
        assert directional.block_circulant_defect == ((0,) * 6,) * 6
        assert directional.source_contrast_a == contrast_a
        assert directional.source_contrast_b == contrast_b
        assert directional.mean_source == (0, 0)
        assert directional.contrast_generator_real is None
        assert directional.contrast_generator_imag_over_sqrt3 is None


def test_affine_closure_admits_the_zero_law_without_inventing_a_phase():
    report = form_geometry.derive_regional_affine_closure(
        (0, 1, 2), ((0, 1, 2),), generator=((0, 0, 0),) * 3, source=(0, 0, 0)
    )
    assert report.all_state_closed
    assert report.mean_generator == report.contrast_generator_real == ((0,),)
    assert report.contrast_generator_imag_over_sqrt3 == ((0,),)
    assert (
        report.mean_source
        == report.source_contrast_a
        == report.source_contrast_b
        == (0,)
    )
    # Declared rational laws retain exact coefficients even outside binary64's
    # range. A tiny directional source is therefore not a successful zero law.
    tiny = Fraction(1, 2**2000)
    directional = form_geometry.derive_regional_affine_closure(
        (0, 1, 2),
        ((0, 1, 2),),
        generator=((0, 0, 0),) * 3,
        source=(tiny, -tiny, 0),
    )
    assert not directional.all_state_closed
    assert directional.source_contrast_a == (2 * tiny,)
    assert directional.source_contrast_b == (0,)


def test_affine_closure_uses_node_identities_and_declared_region_order():
    generator, source = _directed_affine_form_model()
    permutation = (3, 0, 5, 2, 4, 1)
    reordered = tuple(tuple(generator[i][j] for j in permutation) for i in permutation)
    report = form_geometry.derive_regional_affine_closure(
        permutation,
        REGIONS,
        generator=reordered,
        source=tuple(source[i] for i in permutation),
    )
    original = form_geometry.derive_regional_affine_closure(
        tuple(range(6)), REGIONS, generator=generator, source=source
    )
    for field in (
        "all_state_closed",
        "source_contrast_a",
        "source_contrast_b",
        "mean_generator",
        "mean_source",
        "contrast_generator_real",
        "contrast_generator_imag_over_sqrt3",
    ):
        assert getattr(report, field) == getattr(original, field)
    reversed_regions = form_geometry.derive_regional_affine_closure(
        tuple(range(6)), REGIONS[::-1], generator=generator, source=source
    )
    assert reversed_regions.mean_source == (-2, 1)
    assert reversed_regions.mean_generator == ((-1, 1), (2, -2))
    assert reversed_regions.contrast_generator_real == (
        (Fraction(-5, 2), 1),
        (-1, Fraction(-7, 2)),
    )
    changed = [list(row) for row in reordered]
    changed[permutation.index(1)][permutation.index(0)] += Fraction(1, 8)
    rejected = form_geometry.derive_regional_affine_closure(
        permutation,
        REGIONS,
        generator=changed,
        source=tuple(source[i] for i in permutation),
    )
    assert not rejected.all_state_closed
    nonzero_defects = tuple(
        (i, j, value)
        for i, row in enumerate(rejected.block_circulant_defect)
        for j, value in enumerate(row)
        if value
    )
    assert nonzero_defects == ((5, 1, Fraction(1, 8)),)


def test_affine_closure_rejects_malformed_laws_and_unordered_or_incomplete_frames():
    generator, source = _directed_affine_form_model()
    for malformed_generator, malformed_source in (
        (generator[:-1], source),
        (tuple(row[:-1] for row in generator), source),
        (generator, source[:-1]),
        (None, source),
        (generator, {0: 1}),
    ):
        with pytest.raises((TypeError, ValueError)):
            form_geometry.derive_regional_affine_closure(
                tuple(range(6)),
                REGIONS,
                generator=malformed_generator,
                source=malformed_source,
            )
    for invalid in (True, 1 + 0j, float("nan"), float("inf"), "1"):
        for target in ("generator", "source"):
            bad_generator = [list(row) for row in generator]
            bad_source = list(source)
            if target == "generator":
                bad_generator[0][0] = invalid
            else:
                bad_source[0] = invalid
            with pytest.raises((TypeError, ValueError)):
                form_geometry.derive_regional_affine_closure(
                    tuple(range(6)),
                    REGIONS,
                    generator=bad_generator,
                    source=bad_source,
                )
    for nodes, regions in (
        ((), ()),
        ((0, 1, 2, 3, 4, 4), REGIONS),
        (set(range(6)), REGIONS),
        (tuple(range(6)), {REGIONS[0], REGIONS[1]}),
        (tuple(range(6)), ((0, 1, 2),)),
        (tuple(range(6)), ((0, 1, 2), (2, 4, 5))),
        (tuple(range(6)), ((0, 1, 2), (3, 4, 99))),
    ):
        with pytest.raises((TypeError, ValueError)):
            form_geometry.derive_regional_affine_closure(
                nodes, regions, generator=generator, source=source
            )
