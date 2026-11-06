"""Manufactured arrival controls; no original-source response is evaluated."""

from dataclasses import asdict, replace
from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric_inclusion import validated_metric_inclusion_step
from tnfr.research import formed_two_port_arrival as owner
from tnfr.research import formed_two_port_reference as reference_owner
from tnfr.research.artifact_io import encode_exact_tree


@pytest.fixture(autouse=True)
def no_scientific_response(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("arrival controls may not evaluate an actual source or target")
    monkeypatch.setattr(reference_owner.FormedTwoPortReference, "field", forbidden)
    monkeypatch.setattr(reference_owner, "admit_formed_two_port_target", forbidden)
    monkeypatch.setattr(reference_owner, "admit_formed_two_port_acute_exit_prefix", forbidden)


def norm2(row, metric):
    return sum(x*y*metric[i][j] for i, x in enumerate(row)
               for j, y in enumerate(row))


def test_common_weighted_mean_chart_is_exact_and_isometric():
    chart = owner.WeightedMeanChart((2, 3, 2))
    original = (7, -5, 4)
    projected = chart.project(original)
    full = chart.lift(projected)
    mean = Q(sum(d*x for d, x in zip(chart.degrees, original)), chart.mass)
    assert full == tuple(x-mean for x in original)
    assert all(type(x) is Q for x in projected+full)
    assert sum(d*x for d, x in zip(chart.degrees, full)) == 0
    assert chart.project(tuple(x+13 for x in original)) == projected
    assert chart.project(full) == projected
    assert norm2(projected, chart.metric) == sum(d*x*x for d, x in zip(chart.degrees, full))
    assert exact_symmetric_semidefinite(chart.metric, strict=True)


@pytest.mark.parametrize("bad", (True, 0.5, "1"))
def test_chart_rejects_inexact_or_boolean_coordinates(bad):
    chart = owner.WeightedMeanChart((2, 2, 2))
    with pytest.raises((ValueError, TypeError)):
        chart.lift((0, bad))
    with pytest.raises((ValueError, TypeError)):
        chart.project((0, bad, 0))


def test_chart_interval_and_jet_extensions_preserve_formal_coordinates():
    chart = owner.WeightedMeanChart((2, 3, 2))
    intervals = (I(-1, 2), I(1, 3))
    full = chart.lift(intervals)
    assert full[-1].contains(Q(-13, 2)) and full[-1].contains(Q(-1, 2))
    jets = (Jet((I(1), I(2), I(0))), Jet((I(3), I(-1), I(1))))
    lifted = chart.lift(jets)
    for k in range(3):
        assert (sum(d*x.coeffs[k] for d, x in zip(chart.degrees, lifted))).contains(0)
    toy = chart.pullback_flow(lambda values: tuple(-2*x for x in values))
    assert toy(jets) == tuple(-2*x for x in jets)
    with pytest.raises(ValueError, match="mix"):
        chart.lift((1, jets[1]))
    with pytest.raises(ValueError, match="order"):
        chart.lift((jets[0], Jet.constant(0, 1)))


def test_actual_chart_build_is_algebra_only_and_records_all_seventeen_modes():
    model = owner.build_formed_two_port_arrival_model()
    assert model.chart.dimension == 17 and model.chart.mass == 40
    assert model.metric == model.chart.metric
    assert model.domain((I(-100, 100),)*17) == (Q(1),)
    assert exact_symmetric_semidefinite(model.metric, strict=True)
    # A manufactured drift tests adapters without evaluating the scientific law.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reference_owner.FormedTwoPortReference, "field",
                      lambda self, full: tuple(-2*x for x in full))
        row = (Q(1, 100),)*17
        assert model.flow(row) == tuple(-2*x for x in row)
        jets = tuple(Jet.constant(x, 2) for x in row)
        assert model.flow(jets) == tuple(-2*x for x in jets)
        upper = model.field_norm_upper(tuple(I(x) for x in row))
        assert type(upper) is Q
        assert model.forcing_bound(tuple(I(x) for x in row)) == min(Q(1, 7000), upper/20000)


def test_signed_graph_certificate_proves_relative_contraction_without_a_zero_mode():
    chart = owner.WeightedMeanChart((2, 2, 2))
    edges = ((0, 1), (1, 2), (2, 0))
    proof = owner._graph_growth(chart, edges, (1, 1, 1), (I(-1, 1),)*2,
                                (Q(-2), Q(2)), 8)
    assert proof.certified and proof.growth_rate_upper_bound == Q(-3, 2)
    assert proof.stiffness_lower_bound == tuple(tuple(Q(3, 2)*x for x in row)
                                                for row in chart.metric)
    assert exact_symmetric_semidefinite(proof.slack_matrix)
    assert proof.tube == (I(-1, 1),)*2
    assert proof.metric == chart.metric
    assert proof.bisections_performed == 8
    encode_exact_tree(asdict(proof))


def test_signed_graph_negative_edges_and_nonexpanded_failed_bracket():
    chart = owner.WeightedMeanChart((2, 2, 2))
    edges = ((0, 1), (1, 2), (2, 0))
    bad = owner._graph_growth(chart, edges, (-1,)*3, (I(0),)*2, (0, 1), 8)
    assert not bad.certified and bad.growth_rate_upper_bound is None
    assert bad.bisections_performed == 0
    good = owner._graph_growth(chart, edges, (-1,)*3, (I(0),)*2, (0, 2), 8)
    assert good.certified and good.growth_rate_upper_bound == Q(3, 2)
    assert exact_symmetric_semidefinite(good.slack_matrix)


def test_structured_growth_report_consumed_by_shared_forced_step_on_toy_system():
    chart = owner.WeightedMeanChart((2, 2, 2))
    edges = ((0, 1), (1, 2), (2, 0))
    def growth(tube):
        return owner._graph_growth(chart, edges, (1,)*3, tube, (-2, 2), 8)
    result, failed, reason = validated_metric_inclusion_step(
        (Q(1, 10), Q(-1, 10)), Q(1, 100), chart.metric, Q(1, 16),
        lambda state: tuple(-Q(3, 2)*x for x in state), lambda _: (Q(1),),
        forcing_bound=Q(1, 100), growth_certifier=growth, order=8)
    assert result is not None and failed is reason is None
    assert result.external_growth_certificate.growth_rate_upper_bound == Q(-3, 2)
    assert result.endpoint_radius < Q(1, 100)
    assert result.forcing_radius_upper_bound > 0
    assert result.external_growth_certificate.tube == result.growth_proof_tube


@pytest.mark.parametrize("weight", (Q(1), Q(-1)))
def test_actual_support_extreme_constant_cosines_have_a_finite_exact_certificate(weight):
    model = owner.build_formed_two_port_arrival_model()
    proof = owner._graph_growth(model.chart, model.reference.oriented_edges,
                                (weight,)*20, (I(-2, 2),)*17, (Q(-1, 64), Q(2)), 4)
    assert proof.certified
    assert exact_symmetric_semidefinite(proof.slack_matrix)
    if weight == 1:
        assert proof.growth_rate_upper_bound < 0
    else:
        assert proof.growth_rate_upper_bound > 0


def test_growth_adapter_binds_lifted_tube_to_manufactured_angles(monkeypatch):
    model = owner.build_formed_two_port_arrival_model()
    tube = tuple(I(-Q(j, 100), Q(j, 100)) for j in range(17))
    observed = []
    def manufactured_angles(self, lifted):
        observed.append(lifted)
        return (I(0),)*20
    monkeypatch.setattr(reference_owner.FormedTwoPortReference, "angles", manufactured_angles)
    proof = model.certify_growth(tube, bisections=4)
    assert observed == [model.chart.lift(tube)]
    assert proof.certified and proof.growth_rate_upper_bound < 0
    assert proof.edge_cosine_lower_bounds == (Q(1),)*20
    assert proof.tube == tube and proof.metric == model.metric


@pytest.mark.parametrize("rates,steps,weights", (
    ((2, 1), 3, (1,)), ((0, 2), True, (1,)), ((0, 2), 17, (1,)),
    ((0, 2), 3, (True,)), ((0, 2), 3, (Q(-3, 2),)),
))
def test_graph_growth_rejects_malformed_exact_policy(rates, steps, weights):
    with pytest.raises((ValueError, TypeError)):
        owner._graph_growth(owner.WeightedMeanChart((1, 1)), ((0, 1),),
                            weights, (I(0),), rates, steps)


def test_source_bridge_projects_once_and_retains_independent_radius(monkeypatch):
    model = owner.build_formed_two_port_arrival_model()
    # Manufactured admitted-prefix object; no real prefix or field is read.
    center = tuple(Q(j, 19) for j in range(18))
    radius = Q(1, 2**30)
    receipts = ({"path": "manufactured-only", "sha256": "synthetic"},)
    prefix = reference_owner.FormedTwoPortAcuteExitPrefix(center, radius, (Q(1),), receipts)
    calls = []
    def reader(reference, path):
        calls.append((reference, path))
        return prefix
    monkeypatch.setattr(reference_owner, "admit_formed_two_port_acute_exit_prefix", reader)
    source = owner.admit_formed_two_port_arrival_source(model, "synthetic-prefix")
    assert len(calls) == 1 and calls[0][1] == "synthetic-prefix"
    assert source.time == 1 and source.center == model.chart.project(center)
    assert source.radius == radius+Q(1, 45000)
    assert source.prefix_reference_radius == radius
    assert source.prefix_radius_premise_margin == Q(1, 2**26)-radius
    assert source.reference_center_weighted_mean == sum(d*x for d, x in zip(model.chart.degrees, center))/40
    assert source.evidence_receipts == receipts
    encode_exact_tree(asdict(source))
    monkeypatch.setattr(reference_owner, "admit_formed_two_port_acute_exit_prefix",
                        lambda *args: replace(prefix, endpoint_radius=Q(1, 2**26)))
    with pytest.raises(ValueError, match="prefix radius"):
        owner.admit_formed_two_port_arrival_source(model, "synthetic-prefix")


def test_terminal_capture_uses_full_form_and_phase_handoff():
    result = owner._terminal_arithmetic(1, Q(1, 1024), Q(1, 2048), 0, 1, 1)
    assert result.status == "certified_capture"
    assert result.capture_form_upper_bound < Q(1, 8192)
    assert result.capture_excess_upper_bound < Q(1, 648000)
    assert result.nonarrival_storage_margin < 0
    encode_exact_tree(asdict(result))


def test_terminal_rebuilds_cached_target_bounds_and_binds_receipts(monkeypatch):
    model = owner.build_formed_two_port_arrival_model()
    receipts = ({"path": "manufactured-target", "sha256": "synthetic"},)
    retained = reference_owner.FormedTwoPortTarget(
        (I(0),)*18, (I(0),)*18, (), Q(1), receipts, "synthetic", "synthetic")
    target = owner.ArrivalTarget(retained, I(-100), (I(100),)*18, "synthetic", receipts)
    rebuild_calls = []
    def rebuilt(model_argument, retained_argument):
        rebuild_calls.append((model_argument, retained_argument))
        return (I(0),)*18, I(2)
    monkeypatch.setattr(owner, "_target_geometry", rebuilt)
    monkeypatch.setattr(reference_owner.FormedTwoPortReference, "field",
                        lambda self, state: (I(0),)*18)
    monkeypatch.setattr(reference_owner.FormedTwoPortReference, "angles",
                        lambda self, state: (I(0),)*20)
    result = owner.assess_formed_two_port_arrival_terminal(model, target, (Q(1),)*17, 0, 1)
    assert result.status == "certified_nonarrival"
    assert result.target_storage_lower_bound == 2
    assert rebuild_calls == [(model, retained)]
    with pytest.raises(ValueError, match="receipt"):
        owner.assess_formed_two_port_arrival_terminal(
            model, replace(target, evidence_receipts=()), (Q(1),)*17, 0, 1)


def test_terminal_exclusion_retains_positive_form_storage_and_strict_target_gap():
    result = owner._terminal_arithmetic(2, 1, Q(1, 100), Q(1, 10), 1, Q(11, 10))
    assert result.status == "certified_nonarrival"
    expected = (Q(1, 10)+Q(2, 100))*(1+Q(1, 20000))
    assert result.form_storage_upper_bound == expected**2/200000
    assert result.total_storage_upper_bound > result.phase_potential_upper_bound
    equal = owner._terminal_arithmetic(2, 1, 0, 0, 1, 1)
    assert equal.status == "unavailable" and equal.nonarrival_storage_margin == 0
    below = owner._terminal_arithmetic(2, 1, 0, 0, 1, Q(9, 10))
    assert below.status == "unavailable"


def test_mutually_inconsistent_terminal_claims_are_rejected():
    with pytest.raises(ValueError, match="contradictory"):
        owner._terminal_arithmetic(1, 0, 0, 0, 0, 1)


@pytest.mark.parametrize("arguments", (
    (0, 1, 0, 0, 1, 1), (1, 1, -1, 0, 1, 1), (1, 1, 2, 0, 1, 1),
    (1, 1, 0, -1, 1, 1), (1, 1, 0, 0, -1, 1), (1, 1, 0, 0.0, 1, 1),
    (True, 1, 0, 0, 1, 1),
))
def test_terminal_bound_admission_rejects_invalid_primitives(arguments):
    with pytest.raises(ValueError):
        owner._terminal_arithmetic(*arguments)
