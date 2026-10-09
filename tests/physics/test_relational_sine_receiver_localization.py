"""Full nonlinear receiver-localization controls, without response trajectories.

The weighted functional selects an asymptotic receiver state on its admitted
source class. It does not bound transient receiver potential or donor identity.
"""

from dataclasses import replace
from fractions import Fraction as Q

import pytest
from mpmath import mp

from tests.physics.test_relational_sine_formation import _explicit, _report, _support
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def algebra():
    s = pytest.importorskip("sympy")
    edges, neighbors = _support()
    incidence = s.zeros(11, 12)
    for column, (i, j) in enumerate(edges):
        incidence[i, column], incidence[j, column] = 1, -1
    capacities = [
        s.Rational(3, 2) if i == 6 else s.Rational(1, 2) if i == 9 else 1
        for i in range(11)
    ]
    mobility = s.diag(
        *(nu / s.Integer(len(row)) for nu, row in zip(capacities, neighbors))
    )
    donor, receiver = s.zeros(11), s.zeros(11)
    for region, nodes, port in ((donor, range(5), 0), (receiver, range(5, 10), 5)):
        for node in nodes:
            region[node, node] = 1
            region[port, node] -= 1
    bridge = s.eye(11) - donor - receiver
    weights = s.diag(*([1] * 5 + [s.Rational(7, 5)] * 5 + [s.Rational(7, 6)] * 2))
    regional_map = donor + s.Rational(7, 5) * receiver + s.Rational(7, 6) * bridge
    return (
        s,
        edges,
        neighbors,
        incidence,
        mobility,
        (donor, receiver, bridge),
        weights,
        regional_map,
    )


def test_regional_gradient_projection_retains_both_live_bridges(algebra):
    s, _, _, incidence, _, projections, weights, regional_map = algebra
    assert sum(projections, s.zeros(11)) == s.eye(11)
    for first in range(3):
        for second in range(3):
            expected = projections[first] if first == second else s.zeros(11)
            assert projections[first] * projections[second] == expected
    assert regional_map * incidence == incidence * weights
    current = incidence * s.Matrix(s.symbols("sine0:12"))
    receiver_current = projections[1] * current
    assert receiver_current == incidence[:, 5:10] * s.Matrix(s.symbols("sine5:10"))
    assert projections[2] * current == incidence[:, 10:] * s.Matrix(
        s.symbols("sine10:12")
    )
    assert sum(current) == 0


def test_weighted_storage_derivative_uses_actual_sine_rows_and_cosine_remainder(
    algebra,
):
    s, edges, neighbors, incidence, k, _, weights, regional_map = algebra
    x = s.Matrix(s.symbols("x0:11", real=True))
    sine = s.Matrix(s.symbols("sine0:12", real=True))
    cosine = s.symbols("cosine0:12", real=True)
    a = s.Symbol("a", positive=True)
    e, h = s.Rational(1, 2), s.Rational(1, 3)
    laplacian = incidence * incidence.T
    q, current = laplacian * x, incidence * sine
    xdot, phasedot = k * (-e * q + a * current), a * k * q
    qdot = s.Matrix(
        [sum(xdot[node] - xdot[j] for j in row) for node, row in enumerate(neighbors)]
    )
    current_dot = s.zeros(11, 1)
    for edge, (i, j) in enumerate(edges):
        edge_rate = cosine[edge] * (phasedot[j] - phasedot[i])
        current_dot[i] += edge_rate
        current_dot[j] -= edge_rate
    weighted_phase_rate = -(incidence * weights * sine).dot(phasedot)
    actual = q.dot(xdot) + weighted_phase_rate
    actual -= h * (qdot.dot(k * current) + q.dot(k * current_dot))
    upper = -e * q.dot(k * q)
    upper += a * q.dot(k * (s.eye(11) - regional_map) * current)
    upper += h * e * q.dot(k * laplacian * k * current)
    upper -= h * a * current.dot(k * laplacian * k * current)
    upper += h * a * q.dot(k * laplacian * k * q)
    velocity = k * q
    remainder = (
        h
        * a
        * sum(
            (1 - cosine[edge]) * (velocity[i] - velocity[j]) ** 2
            for edge, (i, j) in enumerate(edges)
        )
    )
    assert s.expand(upper - actual - remainder) == 0
    # The remainder stays nonnegative at every phase, not only while either
    # ring remains acute or inside its original winding component.
    assert h > 0


def test_exact_projected_dissipation_matrix_is_positive_at_both_pi_endpoints(algebra):
    s, _, _, incidence, k, _, _, regional_map = algebra
    laplacian = incidence * incidence.T
    projection = s.eye(10).col_join(-s.ones(1, 10))
    joint = s.diag(projection, projection)
    a = s.Symbol("a")
    e, h = s.Rational(1, 2), s.Rational(1, 3)
    product = k * laplacian * k
    cross = -(a * k * (s.eye(11) - regional_map) + h * e * product) / 2
    full = (
        (e * k - h * a * product)
        .row_join(cross)
        .col_join(cross.T.row_join(h * a * product))
    )
    reduced = joint.T * full * joint
    assert reduced.shape == (20, 20)
    assert reduced == reduced.T
    lower, upper = s.Rational(7, 44), s.Rational(25, 157)
    for endpoint in (lower, upper):
        matrix = reduced.subs(a, endpoint)
        triangular, diagonal = matrix.LDLdecomposition(hermitian=False)
        assert triangular * diagonal * triangular.T == matrix
        assert all(pivot > s.Rational(1, 10000) for pivot in diagonal.diagonal())
    # Affinity extends the exact endpoint certificates to the full interval.
    t = s.Symbol("t", nonnegative=True)
    interpolated = (1 - t) * reduced.subs(a, lower) + t * reduced.subs(a, upper)
    assert (reduced.subs(a, (1 - t) * lower + t * upper) - interpolated).applyfunc(
        s.expand
    ) == s.zeros(20)
    pi = pi_interval()
    assert Q(7, 44) < 1 / (2 * pi.hi) <= 1 / (2 * pi.lo) < Q(25, 157)


def test_all_nonflat_cycle_critical_geometries_have_at_least_twist_potential():
    s = pytest.importorskip("sympy")
    twist = (25 - 5 * s.sqrt(5)) / 4
    values = []
    for negative_edges in range(6):
        coefficient = 5 - 2 * negative_edges
        # Closure restricts m to k/2 +/- abs(5-2k)/4, strictly inside
        # this finite enclosing integer range. No phase grid is sampled.
        for winding in range(-2, 5):
            angle_ratio = s.Rational(2 * winding - negative_edges, coefficient)
            if abs(angle_ratio) >= s.Rational(1, 2):
                continue
            potential = s.simplify(5 - coefficient * s.cos(s.pi * angle_ratio))
            if potential == 0:
                assert negative_edges == winding == 0
                continue
            assert s.simplify(potential - twist) >= 0
            values.append(potential)
    assert any(value == twist for value in values)
    assert all(value > 0 for value in values)
    # At any full equilibrium, q=0 kills the mixed term and every donor
    # and bridge phase contribution is nonnegative. This excludes receiver
    # saddle limits too, rather than assuming an attracting endpoint.
    assert s.simplify(s.Rational(7, 5) * twist - twist) == (5 - s.sqrt(5)) / 2


def test_report_selects_receiver_consensus_without_selecting_donor_or_transient_state():
    source = _report(1)
    report = source.receiver_localization()
    assert not source.donor_well_retention().receiver_only_targets_excluded
    assert not source.donor_dissipative_capture().receiver_only_targets_excluded
    assert report.source is source
    assert report.status == "certified"
    assert report.receiver_nonflat_equilibria_excluded
    assert report.relative_receiver_consensus_certified
    assert report.initial_form_storage == 1
    assert report.form_storage_coefficient == report.donor_phase_weight == 1
    assert report.mixed_term_coefficient == Q(1, 3)
    assert report.receiver_phase_weight == Q(7, 5)
    assert report.bridge_phase_weight == Q(7, 6)
    assert report.target_minus_initial_margin_bounds.lo > 0
    assert not hasattr(report, "donor_only_target_certified")
    assert not hasattr(report, "receiver_phase_potential_upper_bound")
    assert "no_all_time_receiver_barrier_winding_or_potential_bound" in report.scope
    transfer = source.receiver_transfer()
    assert transfer.receiver_localization == report
    assert type(transfer).__dataclass_fields__["receiver_localization"].default is None
    assert transfer.status == "excluded"
    assert "weighted_receiver_localization" in transfer.exclusion_reasons


def test_localization_uses_actual_full_form_storage_and_preserves_zero_source():
    for source in (
        _report(0),
        _report(Q(1, 2**200)),
        _report(1, profile="balanced"),
        _explicit(donor=(0, Q(1, 4), 0, 0, Q(-1, 4)), hidden=1),
    ):
        report = source.receiver_localization()
        assert report.status == "certified"
        assert report.initial_form_storage == source.initial_form_storage
        assert report.relative_receiver_consensus_certified
    # Odd form is silent on its invariant source, but its full energy still
    # counts for this particular theorem; no even-source budget is substituted.
    odd = _explicit(donor=(0, 1, 0, 0, -1), hidden=0)
    assert odd.silent_donor_subspace
    assert odd.receiver_excitation().even_form_storage == 0
    assert odd.receiver_localization().status == "not_certified"
    # Squaring the radical threshold without its sign condition would
    # incorrectly admit this larger, original-budget source.
    high = _report(2).receiver_localization()
    assert high.initial_form_storage == 4
    assert high.exact_localization_polynomial_margin > 0
    assert high.status == "not_certified"
    assert not high.relative_receiver_consensus_certified


def test_exact_threshold_is_not_decided_by_rounded_display_intervals():
    context = mp.clone()
    context.dps = 120
    threshold = (5 - context.sqrt(5)) / 2
    scale = 2**240
    numerator = int(context.sqrt(threshold) * scale)
    reports = [
        _report(Q(numerator + offset, scale)).receiver_localization()
        for offset in (0, 1)
    ]
    assert reports[0].status == "certified"
    assert reports[1].status == "not_certified"
    for report in reports:
        margin = report.target_minus_initial_margin_bounds
        assert margin.contains(0)
        form = report.initial_form_storage
        assert report.exact_localization_polynomial_margin == (5 - 2 * form) ** 2 - 5
        lower, upper = (
            report.critical_form_storage_bounds.lo,
            report.critical_form_storage_bounds.hi,
        )
        assert context.mpf(lower.numerator) / lower.denominator <= threshold
        assert threshold <= context.mpf(upper.numerator) / upper.denominator


@pytest.mark.parametrize(
    "source",
    (
        _report(1, model=RelationalExchangeModel(2, phase_domain="regular")),
        _report(1, model=RelationalExchangeModel(1, 3, 2, phase_domain="regular")),
        _report(1, contrast=Q(-1, 2)),
        replace(_report(1), law="other_pressure_law"),
    ),
)
def test_unsupported_law_never_inherits_localization_from_the_preparation(source):
    report = source.receiver_localization()
    assert report.status == "unavailable" and report.unavailable_reasons
    assert not report.receiver_nonflat_equilibria_excluded
    assert not report.relative_receiver_consensus_certified
    assert report.initial_form_storage == source.initial_form_storage
    assert report.critical_form_storage_bounds.lo > 0
    assert report.form_storage_coefficient is None
    assert report.mixed_term_coefficient is None
    assert report.donor_phase_weight is None
    assert report.receiver_phase_weight is None
    assert report.bridge_phase_weight is None
    assert report.initial_functional_bounds is None
    assert report.receiver_nonflat_limit_functional_lower_bounds is None
    assert report.target_minus_initial_margin_bounds is None


def test_reader_revalidates_preparation_recomputes_evidence_and_exports(tmp_path):
    source = _report(1)
    report = source.receiver_localization()
    for change in (
        {"initial_epi": (False,) + source.initial_epi[1:]},
        {"capacity": (True,) + source.capacity[1:]},
        {"initial_form_storage": Q(0)},
        {"initial_phase_turns": (Q(0),) * 11},
        {"degrees": (1,) * 11},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(source, **change).receiver_localization()
    stale = replace(source, twist_phase_storage_bounds=I(99)).receiver_localization()
    assert stale.critical_form_storage_bounds == report.critical_form_storage_bounds
    assert stale.initial_functional_bounds == report.initial_functional_bounds
    assert (
        stale.target_minus_initial_margin_bounds
        == report.target_minus_initial_margin_bounds
    )
    assert stale.relative_receiver_consensus_certified
    projected = report.to_dict()
    assert projected["schema"] == "tnfr.relational-sine-receiver-localization.v1"
    assert projected["report"]["receiver_phase_weight"] == {
        "numerator": 7,
        "denominator": 5,
    }
    path = tmp_path / "receiver-localization.json"
    export_to_json(report, path)
    exported = json_loads(path.read_bytes())
    assert exported["report"] == projected["report"]
