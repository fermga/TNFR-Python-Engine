"""Exact no-go for a declared two-boundary Mutation renewal sector.

Freeze doubled C5, unit conductance, contiguous twins, zero internal
contrasts, forms (x,y,z,z,y) in [-1,1], capacities (u,W,W,W,W) with
0<u<W<=VF_MAX, and a coherent phase lift (D,a,0,0,a), 0<=a<=D<=3.
Retain source and both hypothetical endpoints inside the form rails, Gamma0,
default laws/gains, dt1/16, supplied AL/IL/EN context, zero debt/lags/counts,
empty physical history and finite zero old rates/acceleration. First requests
are OZ/IL/IL/IL/IL; second requests are ZHIR/IL/IL/IL/IL, with the acquired
positive secant, associated samples and complete second-call phase renewal.

The budget is exactly two Mutation directions times three first-controller
branches times three second-controller branches. Both calls retain their
actual common-arc premise, including post-Mutation phases consumed by the
second coordinator; there is no source search, source response or
trajectory evaluation. The 18 cells are a relaxation: impossible controller
combinations are harmless. Freeze rational certificates before independent
coefficient/optimization controls. No tolerances or posterior tuning occur.

Multiplicative OZ cannot supply the required secant: its uniform upper bound
is <.037 for positive renewal and <.008 for negative renewal. OZ's separate
tiny-pressure +.1 fallback gives a rate <.076, also below xi. These are ideal
common-lift inequalities with represented non-angle coefficients, not native
binary64 execution or a global impossibility theorem for TNFR/Mutation.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import product

import pytest

from tests.physics import test_native_mutation_renewal as previous

configuration = previous.configuration
no_execution = previous.no_execution
calculus_guard = previous.calculus_guard
renewal_guard = previous.renewal_guard
DT, PI = previous.DT, previous.PI
BRANCHES = ("stable", "transition", "dissonant")
CELLS = tuple(product((-1, 1), BRANCHES, BRANCHES))
DOMAIN = (
    "doubled C5/unit/order0..9/zero contrasts",
    "M=(x,y,z,z,y), source and endpoints in [-1,1]",
    "V=(u,W,W,W,W),0<u<W<=VF_MAX",
    "theta=(D,a,0,0,a),0<=a<=D<=3<pi; post-Mutation common arc",
    "AL/IL/EN supplied; priorIL; ordinal3; zero debt/lags/counts",
    "empty t1 history; finite zero old rates/acceleration",
    "Gamma0; no callbacks/optional math/REMESH; dt1/16",
    "OZ/IL/IL/IL/IL then ZHIR/IL/IL/IL/IL; D2>D1",
    "exact represented coefficients; no binary64 trajectory claim",
    "18 branch/direction cells; no source controls; exact inequalities",
)


def _update_gains(law, old, branch):
    cfg = law["phase_policy"]
    targets = {
        "stable": (cfg["kG_min"], cfg["kL_min"]),
        "transition": (
            (cfg["kG_min"] + cfg["kG_max"]) / 2,
            (cfg["kL_min"] + cfg["kL_max"]) / 2,
        ),
        "dissonant": (cfg["kG_max"], (cfg["kL_min"] + cfg["kL_max"]) / 2),
    }
    return tuple(
        value + cfg["up" if target > value else "down"] * (target - value)
        for value, target in zip(old, targets[branch])
    )


def _quadratic_max(c0, c1, c2, upper):
    """Closed-interval maximum of a strictly concave rational quadratic."""
    assert c2 < 0 and upper >= 0
    vertex = -c1 / (2 * c2)
    point = max(Q(0), min(upper, vertex))
    return point, c0 + c1 * point + c2 * point * point


@pytest.fixture(scope="module")
def certificate(configuration, renewal_guard):
    # Read only configuration/support, not an old prediction or response fixture.
    law = previous.enclosure._law(configuration)
    k, b, w = law["pressure"]
    assert law["stability"] == (w, k, b)
    assert law["form_rails"] == (Q(-1), Q(1))
    assert law["capacity_rails"][0] == 0 and law["tau"] > 2
    assert configuration.graph["OZ_NOISE_MODE"] is False
    assert min(law["lag_limits"]) >= 2
    cut, low = law["thresholds"]["dnfr_hi"], law["thresholds"]["si_lo"]
    xi = Q(configuration.graph["ZHIR_THRESHOLD_XI"])
    f, retain = law["factors"]["OZ_dnfr_factor"], law["factors"]["IL_dnfr_factor"]
    maximum = law["capacity_rails"][1]
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    d = w / PI
    lam = DT * k * maximum * retain
    kappa = cut / (1 - cut / 2)
    r0, r1 = (low - k - b * (1 - cut)) / w, k / (w * PI)
    assert r0 > 0 and 0 < cut < 1 and 0 < shift < 1
    rows = {}
    for direction, first, second in CELLS:
        g1, l1 = _update_gains(law, law["phase_gains"], first)
        g2, l2 = _update_gains(law, (g1, l1), second)
        cfg = law["phase_policy"]
        assert 0 < cfg["up"] < 1 and 0 < cfg["down"] < 1
        # Each update is a convex interpolation between admitted gains and
        # admitted targets, so the runtime gain clamp is inactive here.
        for global_gain, local_gain in ((g1, l1), (g2, l2)):
            assert cfg["kG_min"] <= global_gain <= cfg["kG_max"]
            assert cfg["kL_min"] <= local_gain <= cfg["kL_max"]
        a1, a2 = 1 - g1 - l1, 1 - g2 - l2
        det = a1 * (a1 - l1 / 2) - l1 * l1 / 4
        ca, ct, ch = (a1 - l1 / 2) / det, l1 / (2 * det), a1 / det - 1
        assert det > 0 and min(ca, ct, ch) > 0
        assert a1 >= l1 / 2 and a2 >= l2 / 2 and min(g1, g2, l1, l2) > 0
        if direction > 0:
            # Exact reflection renewal: A2*s>(G2+L2)h+(G2+L2/2)t.
            t0 = a2 * shift / (g2 + l2 / 2)
            slope = (g2 + l2) / (g2 + l2 / 2)
            hmax = a2 * shift / (g2 + l2)
            n0 = k * (kappa + lam)
            nt = d * (kappa / 2 + lam * ca / 2 + ct)
            nh = d * (lam * ct / 2 + ch)
            intercept, gradient = n0 + nt * t0, nh + nt * slope
            polynomial = (
                r0 * intercept,
                r1 * intercept - r0 * gradient,
                -r1 * gradient,
            )
            # A feasible positive first pressure requires a positive numerator;
            # only there is replacing u by its Si upper bound needed. Negative
            # numerators already exclude the acquired positive secant.
            maximizer, numerator = _quadratic_max(*polynomial, hmax)
            umax = maximum * (r0 + r1 * hmax)
            denominator = 1 - lam / 2 - DT * k * f * umax
            upper = f * maximum * numerator / denominator
            assert upper < Q(37, 1000)
        else:
            # If h+t>=s, a negative shift stays in the original phase hull.
            hmax = shift
            umax = maximum * (r0 + r1 * shift)
            denominator = 1 - lam / 2 - DT * k * f * umax
            upper = umax * f * (lam * k + d * (lam * ca / 2 + ct) * shift) / denominator
            polynomial = maximizer = None
            assert upper < Q(8, 1000)
        assert denominator > Q(95, 100) and 0 < umax < Q(4, 5)
        fallback = Q(0.1) * umax
        assert fallback < (Q(76, 1000) if direction > 0 else Q(26, 1000)) < xi
        rows[(direction, first, second)] = dict(
            gains1=(g1, l1),
            gains2=(g2, l2),
            a1=a1,
            a2=a2,
            det=det,
            ca=ca,
            ct=ct,
            ch=ch,
            hmax=hmax,
            umax=umax,
            denominator=denominator,
            polynomial=polynomial,
            maximizer=maximizer,
            upper=upper,
            fallback=fallback,
        )
    digest = sha256(repr((DOMAIN, CELLS, law, xi, shift, rows)).encode()).hexdigest()
    return dict(
        law=law,
        rows=rows,
        xi=xi,
        shift=shift,
        digest=digest,
        lam=lam,
        kappa=kappa,
        r0=r0,
        r1=r1,
    )


def _mv(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _laplacian(vector):
    return tuple(
        (vector[(i - 1) % 5] + vector[(i + 1) % 5]) / 2 - vector[i] for i in range(5)
    )


def _phase_matrix(g, local):
    return tuple(
        tuple(
            (1 - g - local if i == j else local / 2 if (i - j) % 5 in (1, 4) else Q(0))
            for j in range(5)
        )
        for i in range(5)
    )


def test_certificate_is_guarded_finite_and_has_no_source_responses(
    certificate, renewal_guard
):
    targets, forbidden = renewal_guard
    assert len(CELLS) == len(certificate["rows"]) == 18
    assert len(certificate["digest"]) == 64
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert (
        certificate["law"]["stability"][0] > certificate["law"]["thresholds"]["si_hi"]
    )


def test_common_lift_inverse_is_the_fine_coordinator_restriction(certificate):
    # Coefficient columns, not candidate sources or additional trajectory cells.
    for first in BRANCHES:
        row = certificate["rows"][(1, first, "stable")]
        g, local = row["gains1"]
        a = row["a1"]
        matrix = _phase_matrix(g, local)
        for old_h, old_a in ((Q(1), Q(0)), (Q(0), Q(1))):
            fine = _mv(matrix, (old_h + old_a, old_a, Q(0), Q(0), old_a))
            h, t = fine[0] - fine[1], fine[1] - fine[2]
            assert h == (a - local / 2) * old_h + local * old_a / 2
            assert t == local * old_h / 2 + a * old_a
            assert (a * h - local * t / 2) / row["det"] == old_h
            assert (-local * h / 2 + (a - local / 2) * t) / row["det"] == old_a
            assert old_h - h == row["ch"] * h - row["ct"] * t


def test_held_pressure_transport_and_peak_bound_are_coefficient_identities(certificate):
    law = certificate["law"]
    k, _, w = law["pressure"]
    f, retain = law["factors"]["OZ_dnfr_factor"], law["factors"]["IL_dnfr_factor"]
    # On reflected zero-sum pressure, p=(q,-q/2-r,r,r,-q/2-r).
    # Verify independent q/r columns and u/W coefficients of L diag(v*factor)p.
    for q, far in ((Q(1), Q(0)), (Q(0), Q(1))):
        p = (q, -q / 2 - far, far, far, -q / 2 - far)
        assert sum(p) == 0
        low_part = _laplacian((f * p[0], Q(0), Q(0), Q(0), Q(0)))
        high_part = _laplacian((Q(0), *(retain * value for value in p[1:])))
        assert low_part[0] == -f * q
        assert high_part[0] == -retain * (q / 2 + far)
        assert sum(low_part) == sum(high_part) == 0
    for first in BRANCHES:
        row = certificate["rows"][(1, first, "stable")]
        for theta in ((Q(1), Q(0), Q(0), Q(0), Q(0)), (Q(1), Q(1), Q(0), Q(0), Q(1))):
            transported = _mv(_phase_matrix(*row["gains1"]), theta)
            old_h, h = theta[0] - theta[1], transported[0] - transported[1]
            assert (_laplacian(transported)[0] - _laplacian(theta)[0]) * w / PI == (
                old_h - h
            ) * w / PI
    # Max(|q|,|-q/2-r|,|r|)<=q/2+B for 0<=q<=2B and |r|<=B;
    # outside that regime P=q and positive rho=1 already violates the cut.
    cut = law["thresholds"]["dnfr_hi"]
    assert 0 < cut < 1 and certificate["kappa"] * (1 - cut / 2) == cut
    assert k > 0


def test_positive_quadratic_bounds_and_independent_gap_proof(certificate):
    c = certificate
    k, _, w = c["law"]["pressure"]
    f = c["law"]["factors"]["OZ_dnfr_factor"]
    maximum = c["law"]["capacity_rails"][1]
    for first, second in product(BRANCHES, repeat=2):
        row = c["rows"][(1, first, second)]
        c0, c1, c2 = row["polynomial"]
        x = row["maximizer"]
        value = c0 + c1 * x + c2 * x * x
        assert value >= c0 and value >= c0 + c1 * row["hmax"] + c2 * row["hmax"] ** 2
        # Interior vertex and exact completed square certify every point.
        assert 0 < x < row["hmax"] and c1 + 2 * c2 * x == 0
        assert c0 - c2 * x * x == value
        assert 0 < row["upper"] < Q(37, 1000) < c["xi"]
        # Independent, weaker oscillation budget: h+t<T. Compare xi*den
        # against the unseparated quadratic numerator; no shared maximizer.
        g2, l2 = row["gains2"]
        tmax = row["a2"] * c["shift"] / (g2 + l2 / 2)
        n0 = k * (c["kappa"] + c["lam"])
        nt = w / PI * (c["kappa"] / 2 + c["lam"] * row["ca"] / 2 + row["ct"])
        nh = w / PI * (c["lam"] * row["ct"] / 2 + row["ch"])
        b0, b1 = n0 + nt * tmax, nt + nh
        u0, u1 = maximum * c["r0"], maximum * c["r1"]
        gap0 = c["xi"] * (1 - c["lam"] / 2 - DT * k * f * u0) - f * u0 * b0
        gap1 = -c["xi"] * DT * k * f * u1 - f * (u1 * b0 - u0 * b1)
        gap2 = f * u1 * b1
        vertex = max(Q(0), min(tmax, -gap1 / (2 * gap2)))
        assert gap2 > 0
        assert gap0 + gap1 * vertex + gap2 * vertex * vertex > Q(577, 10000)


def test_phase_renewal_condition_and_negative_hull_obstruction(certificate):
    for first, second in product(BRANCHES, repeat=2):
        row = certificate["rows"][(1, first, second)]
        g, local = row["gains2"]
        a = row["a2"]
        matrix = _phase_matrix(g, local)
        # Independent coefficient columns for the exact output diameter.
        for h, t, shift in ((Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1))):
            value = _mv(matrix, (h + t + shift, t, Q(0), Q(0), t))
            assert value[0] >= value[1] >= value[2]
            assert value[0] - value[2] == a * (h + t + shift) + local * t / 2
            assert (
                value[0] - value[2] - h - t
                == a * shift - (g + local) * h - (g + local / 2) * t
            )
        assert all(entry >= 0 for line in matrix for entry in line)
        assert all(sum(line) == 1 - g < 1 for line in matrix)
        # If s<=h+t, lowering only the maximum remains in the old hull;
        # positive rows with common sum 1-g strictly contract its diameter.
        negative = certificate["rows"][(-1, first, second)]
        assert negative["hmax"] == certificate["shift"]
        assert 0 < negative["upper"] < Q(8, 1000) < certificate["xi"]


def test_fallback_and_zero_normalizer_do_not_escape_the_no_go(certificate):
    xi = certificate["xi"]
    # Literal represented +.1 is a different branch at BOTH closed endpoints
    # |p|<=1e-9; it is not the multiplicative pressure transport identity.
    tiny, fallback = Q(1e-9), Q(0.1)
    for p in (-tiny, Q(0), tiny):
        assert abs(p) <= tiny
        for row in certificate["rows"].values():
            assert 0 < fallback * row["umax"] < Q(76, 1000) < xi
    for row in certificate["rows"].values():
        assert row["denominator"] > 0 and row["umax"] < 1
    # P'=0 implies every p'=0. The positive-pressure envelope still holds
    # with rho0=0, and both possible signed-zero shift directions are included.
    assert {direction for direction, _, _ in certificate["rows"]} == {-1, 1}
    assert certificate["kappa"] > 0


def test_finite_protocol_keeps_memory_and_scope_obligations(certificate):
    law = certificate["law"]
    assert law["tau"] > 2 and min(law["lag_limits"]) >= 2
    # Starting at zero, even two successful qualifications cannot write
    # capacity; counts may be 0,1,2. This justifies the same u,W in both calls.
    counts = {0}
    for _ in range(2):
        counts = {0, *(count + 1 for count in counts)}
    assert counts == {0, 1, 2} and max(counts) < law["tau"]
    # The contradiction uses the associated secant of unclipped held flow,
    # not a fabricated history or a capacity-normalized reconstructed pressure.
    assert (
        max(row["fallback"] for row in certificate["rows"].values()) < certificate["xi"]
    )
    assert max(row["upper"] for row in certificate["rows"].values()) < certificate["xi"]
