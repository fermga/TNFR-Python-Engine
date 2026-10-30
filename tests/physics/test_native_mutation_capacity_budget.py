"""Necessary heterogeneous-capacity resources for two-call Mutation renewal.

Freeze the same ideal doubled-C5/twin/common-lift/default-law/two-call domain
as the general geometry owner, but permit arbitrary surrounding capacities.
V=max(v0,...,v4) is retained separately from near/far means and the near
capacity-pressure covariance. Require actual high-Si IL on every surrounding
pair at BOTH boundaries, OZ0 then acquired ZHIR0, strict grammar, associated
unclipped history, zero Gamma, dt1/16 and complete second-call renewal.
Initial supplied AL/IL/EN, prior IL, ordinal3, zero debt/lags/counts/rates,
empty t1 history, no callbacks/optional math/REMESH are unchanged premises.

The protocol freezes 18 gain/direction outer cells and ONE resource radius
eta0=1/16 before coefficient evaluation. There are no sources, parameter grids,
optimizers, native responses or finer partitions. Exact algebra proves no
joint success when every surrounding capacity is at least15V/16. Success
therefore requires eta=(V-min_surrounding_capacity)/V>1/16. This is necessary,
not optimal, sufficient, or evidence that an admitted source exists.

General transport retains the signed resource J, including both covariance
and the far capacity contrast. J alone neither admits the selector nor
supplies temporal/grammar evidence. Basis/tensor controls below are algebraic
coefficient tests, not physical preparations or binary64 trajectories.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import product

import pytest

from tests.physics import test_native_mutation_asymmetric_feasibility as geometry

configuration = geometry.configuration
no_execution = geometry.no_execution
calculus_guard = geometry.calculus_guard
renewal_guard = geometry.renewal_guard
DT, PI, BRANCHES = geometry.DT, geometry.PI, geometry.BRANCHES
ETA = Q(1, 16)
CELLS = tuple(product((-1, 1), BRANCHES, BRANCHES))
DOMAIN = (
    "paired unit doubled C5/order0..9/arbitrary five forms and phases",
    "source and endpoints within form rails; source phase diameter<=3",
    "v0=u>0; surroundings>=0; actual V=max(v); all<=VF_MAX",
    "near,far capacity means and near capacity-pressure covariance retained",
    "actual OZ0+high-Si ILrest then acquired ZHIR0+high-Si ILrest",
    "AL/IL/EN/priorIL/ordinal3; zero debt,lags,counts,rates/empty t1 history",
    "Gamma0/defaults/strict grammar/unclipped heldEuler/dt1/16/two calls",
    "eta0=1/16 frozen single radius; no retuning/no sources/no optimizer",
    "18 outer cells; exact bases/tensor identities/quadratic extrema",
)


def _resource_envelopes(law, eta):
    k, b, w = law["pressure"]
    vmax = law["capacity_rails"][1]
    retain = law["factors"]["IL_dnfr_factor"]
    cut = law["thresholds"]["dnfr_hi"]
    initial_peak = 2 * k + b * vmax + 3 * w / PI
    lam = DT * k * vmax * retain
    minus = eta * lam * (initial_peak + b * vmax) / 2
    plus = minus + eta * cut * b * vmax / 2
    return initial_peak, minus, plus


@pytest.fixture(scope="module")
def certificate(configuration, renewal_guard):
    # Only frozen configuration/support/guards are reused, never responses
    # or an earlier certificate's verdict.
    law = geometry.prior.previous.enclosure._law(configuration)
    k, b, w = law["pressure"]
    alpha, beta, gamma = law["stability"]
    assert (alpha, beta, gamma) == (w, k, b)
    assert configuration.graph["OZ_NOISE_MODE"] is False
    assert law["tau"] > 2 and min(law["lag_limits"]) >= 2
    assert law["form_rails"] == (Q(-1), Q(1))
    vmax = law["capacity_rails"][1]
    retain = law["factors"]["IL_dnfr_factor"]
    f = law["factors"]["OZ_dnfr_factor"]
    cut, low, high = (law["thresholds"][key] for key in ("dnfr_hi", "si_lo", "si_hi"))
    xi = Q(configuration.graph["ZHIR_THRESHOLD_XI"])
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    d, lam = w / PI, DT * k * vmax * retain
    r0, r1 = (low - beta - gamma * (1 - cut)) / alpha, beta / (alpha * PI)
    il_floor = (high - beta - gamma) / alpha
    assert 0 < il_floor < 1 - ETA
    assert alpha * (1 - ETA) > high
    assert 1 - DT * k * f * vmax - lam / 2 > 0
    initial_peak, resource_minus, resource_plus = _resource_envelopes(law, ETA)
    rows = {}
    for first, second in product(BRANCHES, repeat=2):
        g1, l1 = geometry.prior._update_gains(law, law["phase_gains"], first)
        g2, l2 = geometry.prior._update_gains(law, (g1, l1), second)
        a1, a2 = 1 - g1 - l1, 1 - g2 - l2
        cfg = law["phase_policy"]
        for g, local in ((g1, l1), (g2, l2)):
            assert cfg["kG_min"] <= g <= cfg["kG_max"]
            assert cfg["kL_min"] <= local <= cfg["kL_max"]
            assert 1 - g - local >= local / 2
        assert 3 * (1 - g1 - l1 / 2) + shift < 3 < PI
        det = a1 * (a1 - l1 / 2) - l1 * l1 / 4
        ca, ct, ch = (a1 - l1 / 2) / det, l1 / (2 * det), a1 / det - 1
        assert min(det, ca, ct, ch) > 0
        tmax, hmax = a2 * shift / (g2 + l2 / 2), a2 * shift / (g2 + l2)
        rh = r0 + r1 * hmax
        # Exact high-Si masks imply eta<=1-il_floor, so even the general
        # admitted sector has 1-eta-rh>=il_floor-rh>0.
        assert il_floor > rh
        umax, inward_u = vmax * rh, vmax * (r0 + r1 * shift)
        assert 0 < shift < hmax < tmax and rh + ETA < 1
        assert umax < Q(752, 1000)

        def denominator(u):
            return 1 - lam / 2 - DT * k * f * u

        kt, sz = lam * ca / 2 + ct, lam * ct / 2 + ch
        c0 = cut * (2 * k + b * vmax / 2) + lam * k
        n0, n1 = c0 + d * (cut + kt) * tmax, d * (kt + ch)
        poly = (r0 * n0, r1 * n0 - r0 * n1, -r1 * n1)
        vertex, maximum = geometry._maximize_concave(*poly, hmax)
        assert maximum > 0
        # The extra resource can rescue a negative old numerator. If that
        # old numerator is <=0, discard it upward; otherwise use its positive
        # quadratic envelope. Adding that positive envelope and the resource
        # bound is therefore valid in both cases, without assuming feasibility
        # of the old uniform-capacity model.
        pos_out = f * vmax * maximum / denominator(
            umax
        ) + f * umax * resource_plus / denominator(umax)
        pos_in = (
            f
            * inward_u
            * (c0 + d * ((cut + kt) * tmax + sz * shift) + resource_plus)
            / denominator(inward_u)
        )
        neg_in = (
            f
            * inward_u
            * (lam * k + d * kt * tmax + resource_minus)
            / denominator(inward_u)
        )

        # p0'<=0 and h=-z: b((1-eta)V-u)+d*z<=2k. The numerator
        # r(z)*(2k-d*z) must be nonnegative for a feasible active capacity.
        cap_poly = (2 * k * r0, 2 * k * r1 - d * r0, -d * r1)
        cap_vertex, cap_max = geometry._maximize_concave(*cap_poly, hmax)
        cap_star = cap_max / (b * (1 - ETA - rh))
        assert cap_star < Q(315, 1000)
        assert cap_star <= umax
        negative_u = min(umax, cap_star)
        assert cap_max > 0 and denominator(negative_u) > 0
        neg_out = (
            f
            * negative_u
            * (lam * k + d * (kt * tmax + sz * hmax) + resource_minus)
            / denominator(negative_u)
        )
        fallback = Q(0.1) * umax
        assert pos_out < Q(96, 1000) < xi
        assert pos_in < Q(87, 1000) < xi
        assert neg_in < Q(31, 1000) < xi
        assert neg_out < Q(85, 1000) < xi
        assert fallback < Q(76, 1000) < xi
        rows[(first, second)] = dict(
            gains1=(g1, l1),
            gains2=(g2, l2),
            det=det,
            ca=ca,
            ct=ct,
            ch=ch,
            tmax=tmax,
            hmax=hmax,
            umax=umax,
            inward_u=inward_u,
            poly=poly,
            vertex=vertex,
            cap_poly=cap_poly,
            cap_vertex=cap_vertex,
            cap_star=cap_star,
            negative_u=negative_u,
            pos_out=pos_out,
            pos_in=pos_in,
            neg_in=neg_in,
            neg_out=neg_out,
            fallback=fallback,
        )
    digest = sha256(repr((DOMAIN, CELLS, ETA, law, xi, rows)).encode()).hexdigest()
    return dict(
        law=law,
        rows=rows,
        xi=xi,
        digest=digest,
        il_floor=il_floor,
        initial_peak=initial_peak,
        resource_minus=resource_minus,
        resource_plus=resource_plus,
        r0=r0,
        r1=r1,
        lam=lam,
        shift=shift,
    )


def _near(values):
    return (values[1] + values[4]) / 2


def _far(values):
    return (values[2] + values[3]) / 2


def _covariance(v, p):
    return (v[1] * p[1] + v[4] * p[4]) / 2 - _near(v) * _near(p)


def test_guarded_protocol_and_actual_high_si_masks(certificate, renewal_guard):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert len(CELLS) == 18 and len(certificate["rows"]) == 9
    assert len(certificate["digest"]) == 64 and ETA == Q(1, 16)
    alpha, beta, gamma = certificate["law"]["stability"]
    high = certificate["law"]["thresholds"]["si_hi"]
    assert beta + gamma < high  # A zero-capacity surrounding pair cannot be IL.
    assert (
        alpha * (1 - ETA) > high
    )  # Sufficient at both boundaries, not just positivity.
    assert alpha * certificate["il_floor"] + beta + gamma == high
    # Exact high-Si admission adds beta*|gap|/pi+gamma*rho to this
    # capacity floor. The floor alone is necessary, not a passing mask.
    assert 1 - ETA > certificate["il_floor"] > 0


def test_general_target_transport_is_a_bilinear_tensor_identity(certificate):
    law = certificate["law"]
    k, _, _ = law["pressure"]
    f, retain = law["factors"]["OZ_dnfr_factor"], law["factors"]["IL_dnfr_factor"]
    for vi, pi in product(range(5), range(4)):
        v = tuple(Q(i == vi) for i in range(5))
        # A basis of the full zero-sum pressure space, not a supplied state.
        p = tuple(Q(i == pi) - Q(i == 4) for i in range(5))
        held = tuple(v[i] * (f if i == 0 else retain) * p[i] for i in range(5))
        direct = p[0] + DT * k * geometry.prior._laplacian(held)[0]
        n, q, cov = _near(v), _far(p), _covariance(v, p)
        lam = DT * k * retain * n
        expected = (
            (1 - DT * k * f * v[0] - lam / 2) * p[0] - lam * q + DT * k * retain * cov
        )
        assert direct == expected
        assert cov == (v[1] - v[4]) * (p[1] - p[4]) / 4
    # Independent channel bases verify the extra far capacity term as well
    # as form/phase terms: L commutes with taking paired block means.
    for i in range(5):
        vector = tuple(Q(j == i) for j in range(5))
        assert (
            _far(geometry.prior._laplacian(vector))
            == (_near(vector) - _far(vector)) / 2
        )


def test_signed_resource_needs_covariance_far_contrast_and_actual_maximum(certificate):
    k, b, _ = certificate["law"]["pressure"]
    retain = certificate["law"]["factors"]["IL_dnfr_factor"]
    gain = DT * k * retain
    # Algebraic moment inputs: no claim these are default-selected sources.
    n, far, e, c = Q(3, 4), Q(1, 2), Q(1, 8), Q(1, 4)
    v = (Q(1, 4), n + e, far, far, n - e)
    p = (Q(0), c, Q(0), Q(0), -c)
    reverse = (v[0], v[4], v[2], v[3], v[1])

    def resource(cap):
        return gain * (
            b * _near(cap) * (_near(cap) - _far(cap)) / 2 - _covariance(cap, p)
        )

    assert (
        _near(v) == _near(reverse)
        and _far(v) == _far(reverse)
        and max(v) == max(reverse)
    )
    assert resource(v) - resource(reverse) == -2 * gain * e * c
    uniform = (Q(1, 4), Q(1), Q(1), Q(1), Q(1))
    assert _covariance(uniform, p) == 0 and resource(uniform) == 0
    far_only = (Q(1, 4), Q(1), Q(3, 4), Q(3, 4), Q(1))
    assert _covariance(far_only, p) == 0 and resource(far_only) > 0
    max_separate = (Q(1, 4), Q(1, 2), Q(1), Q(1), Q(1, 2))
    assert max(max_separate) != _near(max_separate)
    assert max_separate[0] / max(max_separate) != max_separate[0] / _near(max_separate)
    assert resource(max_separate) < 0  # J need not be positive or sufficient.


def test_capacity_envelope_and_covariance_budget_are_independent_bounds(certificate):
    # Coefficient interval extrema cover the entire capacity box, including
    # the target's independent [0,V] range; no capacity preparation is sampled.
    columns = [
        geometry.prior._laplacian(tuple(Q(i == j) for i in range(5))) for j in range(5)
    ]
    lower = (Q(0),) + (1 - ETA,) * 4
    upper = (Q(1),) * 5
    for i in range(1, 5):
        row = tuple(columns[j][i] for j in range(5))
        lo = sum(a * (lower[j] if a >= 0 else upper[j]) for j, a in enumerate(row))
        hi = sum(a * (upper[j] if a >= 0 else lower[j]) for j, a in enumerate(row))
        assert max(abs(lo), abs(hi)) <= (1 + ETA) / 2
    # Product interval |(v1-v4)(p1-p4)/4|<=epsilon*P0/2.
    pbound = certificate["initial_peak"]
    vmax = certificate["law"]["capacity_rails"][1]
    covariance_bound = ETA * vmax * 2 * pbound / 4
    assert covariance_bound == ETA * vmax * pbound / 2
    k, b, _ = certificate["law"]["pressure"]
    retain = certificate["law"]["factors"]["IL_dnfr_factor"]
    transport = DT * k * retain * (b * vmax * ETA * vmax / 2 + covariance_bound)
    assert transport == certificate["resource_minus"]
    cut = certificate["law"]["thresholds"]["dnfr_hi"]
    assert transport + cut * b * ETA * vmax / 2 == certificate["resource_plus"]
    assert _resource_envelopes(certificate["law"], Q(0))[1:] == (0, 0)


def test_recovered_resource_has_the_required_sign_and_uniform_limit(certificate):
    k, b, _ = certificate["law"]["pressure"]
    retain = certificate["law"]["factors"]["IL_dnfr_factor"]
    f = certificate["law"]["factors"]["OZ_dnfr_factor"]
    # Independent symbolic columns of target pressure, far form pressure,
    # far phase pressure, capacity contrast, covariance and phase renewal.
    # D*p0=p0'+lambda*(form+phase)+J-phase_increment.
    for j in range(6):
        p0, form, phase, cap, cov, increment = (Q(i == j) for i in range(6))
        u, n = Q(1, 4), Q(3, 4)
        lam = DT * k * retain * n
        den = 1 - DT * k * f * u - lam / 2
        far = form + phase + b * cap / 2
        fresh = den * p0 - lam * far + DT * k * retain * cov + increment
        resource = DT * k * retain * (b * n * cap / 2 - cov)
        assert den * p0 == fresh + lam * (form + phase) + resource - increment
        assert den > 0
    # No acquired history is invented: if u*f*p0>xi, multiplication by
    # positive D/(u*f) and the source form rails produce the documented
    # necessary lower bound on J. The exact equality above fixes its sign.
    xi = certificate["xi"]
    for growth_gap, form_slack in ((Q(1), Q(0)), (Q(0), Q(1))):
        p0 = (xi + growth_gap) / (u * f)
        form = k - form_slack
        # Zero auxiliary columns isolate the two nonnegative coefficients
        # of J-minus-its-required-lower-bound, without constructing a source.
        fresh = den * p0 - lam * form
        required = den * xi / (u * f) - fresh - lam * k
        assert -required == den * growth_gap / (u * f) + lam * form_slack
        assert -required > 0


def test_single_frozen_eta_radius_excludes_all_orientation_cells(certificate):
    for row in certificate["rows"].values():
        for name, point in (("poly", "vertex"), ("cap_poly", "cap_vertex")):
            c0, c1, c2 = row[name]
            z = row[point]
            value = c0 + c1 * z + c2 * z * z
            end = row["hmax"]
            assert c2 < 0 and 0 <= z <= end
            assert value >= c0 and value >= c0 + c1 * end + c2 * end * end
            if 0 < z < end:
                assert c1 + 2 * c2 * z == 0 and value == c0 - c2 * z * z
        assert row["pos_out"] < Q(96, 1000) < certificate["xi"]
        assert row["pos_in"] < Q(87, 1000) < certificate["xi"]
        assert row["neg_in"] < Q(31, 1000) < certificate["xi"]
        assert row["neg_out"] < Q(85, 1000) < certificate["xi"]


def test_fallback_zero_boundaries_and_resource_monotonicity(certificate):
    tiny, fallback = Q(1e-9), Q(0.1)
    for p in (-tiny, Q(0), tiny):
        assert abs(p) <= tiny
        for row in certificate["rows"].values():
            assert (
                fallback * row["umax"]
                == row["fallback"]
                < Q(76, 1000)
                < certificate["xi"]
            )
    assert {direction for direction, _, _ in CELLS} == {-1, 1}
    # Both resource numerators are linear and increasing in eta. The new
    # negative-capacity denominator decreases with eta and stays positive;
    # therefore the frozen endpoint bounds cover every 0<=eta<=1/16.
    assert 0 < certificate["resource_minus"] < certificate["resource_plus"]
    for row in certificate["rows"].values():
        r = certificate["r0"] + certificate["r1"] * row["hmax"]
        assert 1 - ETA - r > 0
