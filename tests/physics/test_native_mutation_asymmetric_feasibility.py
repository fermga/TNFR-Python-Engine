"""Joint Mutation exclusion without reflected form or phase assumptions.

Freeze doubled C5, unit edges, contiguous identical twins, five arbitrary
forms in [-1,1], capacities (u,W,W,W,W), 0<u<W<=VF_MAX, and five paired
primitive phases in a common lift of diameter <=3. The target is not fixed
at an extremum. Source and both hypothetical flow endpoints remain within
the form rails; post-Mutation phases retain their common-arc premise.
Use Gamma0, default factors/thresholds/gains, dt1/16, supplied AL/IL/EN,
prior IL, zero debt/lags/counts, empty t1 history and zero old rates/acceleration.
The required calls are OZ/IL/IL/IL/IL then ZHIR/IL/IL/IL/IL, acquiring the
strict positive secant and increasing complete-call phase diameter.

The frozen budget has 18 gain/direction outer cells and two orientations of
the local gap. Coarse rational inequalities close them all. The reserved
7200 extrema/peak/local-gap regions and any LP solver remain UNUSED. No
source, trajectory or old response is evaluated. Coefficient-basis and
quadratic controls below are algebraic tests, not new preparations.

Acquisition rates are <.091/.089 for positive-shift outward/inward gaps and
<.019/.070 for negative-shift inward/outward gaps. The separate OZ fallback
rate is <.076. All contradict xi=.1. These ideal common-lift conclusions
retain represented coefficients and do not assert binary64 equivariance,
impossibility under other support/laws/histories, or physical identification.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import product

import pytest

from tests.physics import test_native_mutation_joint_feasibility as prior

configuration = prior.configuration
no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard
DT, PI, BRANCHES = prior.DT, prior.PI, prior.BRANCHES
CELLS = tuple(product((-1, 1), BRANCHES, BRANCHES))
DOMAIN = (
    "unit doubled C5/order0..9/identical twins",
    "arbitrary five forms; source/endpoints in [-1,1]",
    "u,W,W,W,W;0<u<W<=VF_MAX; no capacity writes within two calls",
    "arbitrary five lifted phases; diameter<=3; post-Mutation common arc",
    "supplied AL/IL/EN/priorIL/ordinal3; debt,lags,counts zero",
    "empty t1 history; zero old rate/acceleration; Gamma0; dt1/16",
    "OZ then acquired ZHIR at0; others IL; complete second renewal",
    "18 outer cells; two local-gap orientations; exact arithmetic",
    "7200 reserved fine regions UNUSED; no LP/no sources/no response",
)


def _maximize_concave(c0, c1, c2, end):
    assert c2 < 0 and end >= 0
    x = max(Q(0), min(end, -c1 / (2 * c2)))
    return x, c0 + c1 * x + c2 * x * x


@pytest.fixture(scope="module")
def certificate(configuration, renewal_guard):
    # Only shared configuration/support and guards, never prior certificates.
    law = prior.previous.enclosure._law(configuration)
    k, b, w = law["pressure"]
    assert law["stability"] == (w, k, b)
    assert law["form_rails"] == (Q(-1), Q(1))
    assert law["tau"] > 2 and min(law["lag_limits"]) >= 2
    assert configuration.graph["OZ_NOISE_MODE"] is False
    f = law["factors"]["OZ_dnfr_factor"]
    retain = law["factors"]["IL_dnfr_factor"]
    maximum = law["capacity_rails"][1]
    cut = law["thresholds"]["dnfr_hi"]
    low = law["thresholds"]["si_lo"]
    xi = Q(configuration.graph["ZHIR_THRESHOLD_XI"])
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    d, lam = w / PI, DT * k * maximum * retain
    r0, r1 = (low - k - b * (1 - cut)) / w, k / (w * PI)
    assert r0 > 0 and 0 < cut < 1 and shift > 0
    rows = {}
    for first, second in product(BRANCHES, repeat=2):
        gains1 = prior._update_gains(law, law["phase_gains"], first)
        gains2 = prior._update_gains(law, gains1, second)
        g1, l1 = gains1
        g2, l2 = gains2
        a1, a2 = 1 - g1 - l1, 1 - g2 - l2
        # The first common-lift contraction plus either Mutation shift keeps
        # the next coordinator's source inside a strict common arc as well.
        assert 3 * (1 - g1 - l1 / 2) + shift < 3 < PI
        cfg = law["phase_policy"]
        assert 0 < cfg["up"] < 1 and 0 < cfg["down"] < 1
        for g, local in (gains1, gains2):
            assert cfg["kG_min"] <= g <= cfg["kG_max"]
            assert cfg["kL_min"] <= local <= cfg["kL_max"]
            assert 1 - g - local >= local / 2
        det = a1 * (a1 - l1 / 2) - l1 * l1 / 4
        ca, ct, ch = (a1 - l1 / 2) / det, l1 / (2 * det), a1 / det - 1
        assert min(det, ca, ct, ch) > 0
        diameter = a2 * shift / (g2 + l2 / 2)
        hmax = a2 * shift / (g2 + l2)
        assert shift < hmax < diameter

        def radius(h):
            return r0 + r1 * h

        def capacity(h):
            return maximum * radius(h)

        def denominator(u):
            return 1 - lam / 2 - DT * k * f * u

        assert radius(hmax) < 1 and denominator(capacity(hmax)) > Q(95, 100)
        assert capacity(hmax) < Q(752, 1000)

        # Full non-target peak: form row l1 norm2, capacity term <=b*W/2,
        # and phase row bounded by the actual incoming diameter. A ZHIR
        # ratio<1 cannot have a strictly positive target-owned pressure peak.
        peak_constant = cut * (2 * k + b * maximum / 2)
        n0 = peak_constant + lam * k + d * (cut + lam * ca / 2 + ct) * diameter
        n1 = d * (lam * ca / 2 + ct + ch)
        poly = (r0 * n0, r1 * n0 - r0 * n1, -r1 * n1)
        vertex, product_max = _maximize_concave(*poly, hmax)
        pos_out = f * maximum * product_max / denominator(capacity(hmax))

        # For h=-z<0, hull escape gives z<s. The deliberately loose
        # t<=D+z and a<=ca*D+(ca+ct)z keep the positive ct*z term.
        pos_in_n = n0 + d * (lam * (ca + ct) / 2 + ct + ch) * shift
        pos_in = f * capacity(shift) * pos_in_n / denominator(capacity(shift))

        # Negative shift with h>=0: h<s, p0'<=0; omit the peak term.
        neg_in_n0 = lam * k + d * (lam * ca / 2 + ct) * diameter
        neg_poly = (r0 * neg_in_n0, r1 * neg_in_n0 - r0 * n1, -r1 * n1)
        neg_vertex, neg_product = _maximize_concave(*neg_poly, shift)
        neg_in = f * maximum * neg_product / denominator(capacity(shift))

        # Negative shift with h=-z<=0: fresh negative pressure requires
        # b(W-u)+d*z<=2k. Combined with u/W<=r(z), this bounds capacity
        # before using the multiplicative pressure transport identity.
        cap_poly = (2 * k * r0, 2 * k * r1 - d * r0, -d * r1)
        cap_vertex, cap_product = _maximize_concave(*cap_poly, hmax)
        capacity_star = cap_product / (b * (1 - radius(hmax)))
        assert capacity_star < Q(292, 1000)
        negative_capacity = min(capacity(hmax), capacity_star)
        # Escape below the old hull implies t<z+s, independently of which
        # coordinator row realizes the renewed minimum.
        neg_out_n = lam * k + d * (
            (lam * ca / 2 + ct) * shift + (lam * (ca + ct) / 2 + ct + ch) * hmax
        )
        neg_out = f * negative_capacity * neg_out_n / denominator(negative_capacity)
        fallback = Q(0.1) * capacity(hmax)
        assert pos_out < Q(91, 1000) < xi
        assert pos_in < Q(89, 1000) < xi
        assert neg_in < Q(19, 1000) < xi
        assert neg_out < Q(70, 1000) < xi
        assert fallback < Q(76, 1000) < xi
        rows[(first, second)] = dict(
            gains1=gains1,
            gains2=gains2,
            a1=a1,
            a2=a2,
            det=det,
            ca=ca,
            ct=ct,
            ch=ch,
            diameter=diameter,
            hmax=hmax,
            poly=poly,
            vertex=vertex,
            neg_poly=neg_poly,
            neg_vertex=neg_vertex,
            cap_poly=cap_poly,
            cap_vertex=cap_vertex,
            negative_capacity=negative_capacity,
            capacity_max=capacity(hmax),
            pos_out=pos_out,
            pos_in=pos_in,
            neg_in=neg_in,
            neg_out=neg_out,
            fallback=fallback,
        )
    digest = sha256(repr((DOMAIN, CELLS, law, xi, shift, rows)).encode()).hexdigest()
    return dict(
        law=law, rows=rows, xi=xi, shift=shift, digest=digest, lam=lam, r0=r0, r1=r1
    )


def _group(values):
    return values[0], (values[1] + values[4]) / 2, (values[2] + values[3]) / 2


def _contrast(values):
    return (values[1] - values[4]) / 2, (values[2] - values[3]) / 2


def test_guarded_protocol_closes_before_reserved_region_solver(
    certificate, renewal_guard
):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert len(CELLS) == 18 and len(certificate["rows"]) == 9
    assert len(certificate["digest"]) == 64
    law = certificate["law"]
    assert law["stability"][0] > law["thresholds"]["si_hi"]
    # At most two count increments cannot cause capacity adaptation.
    counts = {0}
    for _ in range(2):
        counts = {0, *(n + 1 for n in counts)}
    assert max(counts) < law["tau"]


def test_general_phase_basis_retains_group_inverse_and_antisymmetry(certificate):
    basis = tuple(tuple(Q(i == j) for i in range(5)) for j in range(5))
    for first in BRANCHES:
        row = certificate["rows"][(first, "stable")]
        matrix = prior._phase_matrix(*row["gains1"])
        a1, local = row["a1"], row["gains1"][1]
        for theta in basis:
            advanced = prior._mv(matrix, theta)
            x, near, far = _group(theta)
            y, near1, far1 = _group(advanced)
            old_h, old_a, h, t = x - near, near - far, y - near1, near1 - far1
            assert h == (a1 - local / 2) * old_h + local * old_a / 2
            assert t == local * old_h / 2 + a1 * old_a
            assert old_h - h == row["ch"] * h - row["ct"] * t
            assert old_a == row["ca"] * t - row["ct"] * h
            p, q = _contrast(theta)
            assert _contrast(advanced) == (
                a1 * p + local * q / 2,
                local * p / 2 + (a1 - local / 2) * q,
            )


def test_general_target_transport_and_full_peak_cannot_discard_contrasts(certificate):
    law = certificate["law"]
    k, _, _ = law["pressure"]
    f, retain = law["factors"]["OZ_dnfr_factor"], law["factors"]["IL_dnfr_factor"]
    # Independent q,far,anti-near,anti-far coefficient columns in the full
    # zero-sum pressure subspace, not evaluations of four physical sources.
    for j in range(4):
        q, far, r, s = (Q(i == j) for i in range(4))
        near = -q / 2 - far
        pressure = (q, near + r, far + s, far - s, near - r)
        low = prior._laplacian((f * q, Q(0), Q(0), Q(0), Q(0)))
        high = prior._laplacian((Q(0), *(retain * p for p in pressure[1:])))
        assert sum(pressure) == sum(low) == sum(high) == 0
        assert low[0] == -f * q and high[0] == -retain * (q / 2 + far)
        mean = _group(pressure)
        anti = _contrast(pressure)
        assert max(map(abs, pressure)) == max(
            abs(mean[0]), abs(mean[1]) + abs(anti[0]), abs(mean[2]) + abs(anti[1])
        )
    # Pure antisymmetry has zero grouped means but nonzero pressure peak.
    delta = (Q(0), Q(1), Q(0), Q(0), Q(-1))
    pressure = tuple(k * p for p in prior._laplacian(delta))
    assert _group(pressure) == (0, 0, 0) and max(map(abs, pressure)) > 0
    assert _contrast(prior._laplacian(delta)) == (Q(-1), Q(1, 2))
    # All form rows have l1 norm2; non-target capacity rows touch pair0
    # with coefficient at most1/2. These bound the full fine peak.
    columns = [prior._laplacian(tuple(Q(i == j) for i in range(5))) for j in range(5)]
    assert all(sum(abs(columns[j][i]) for j in range(5)) == 2 for i in range(5))
    assert max(abs(columns[0][i]) for i in range(1, 5)) == Q(1, 2)


def test_renewal_geometry_bounds_the_target_gap_without_reflection(certificate):
    for row in certificate["rows"].values():
        g, local = row["gains2"]
        a = row["a2"]
        matrix = prior._phase_matrix(g, local)
        impulse = tuple(line[0] for line in matrix)
        norm = max(
            sum(abs(x - y) for x, y in zip(first, second)) / 2
            for first in matrix
            for second in matrix
        )
        assert norm == 1 - g - local / 2
        assert sum(impulse[i] > impulse[j] for i in range(5) for j in range(5)) == 8
        # Exact affine extrema on the normalized phase cube certify the two
        # target-row bounds. Translation and positive scaling give any hull.
        for bits in product((Q(0), Q(1)), repeat=5):
            h = bits[0] - (bits[1] + bits[4]) / 2
            value = prior._mv(matrix, bits)[0]
            assert value <= (1 - g) - local * h
            assert value <= (1 - g) + a * h
            assert prior._mv(matrix, tuple(-x for x in bits)) == tuple(
                -x for x in prior._mv(matrix, bits)
            )
        assert (
            local * certificate["shift"] / (2 * (g + local / 2)) < certificate["shift"]
        )
        assert a * certificate["shift"] / (g + local) == row["hmax"]
        # Non-target renewed maxima require D<s. Target maxima give
        # h<hmax for h>=0, or -h<s for h<0; mirror for negative shifts.
        assert certificate["shift"] < row["hmax"] < row["diameter"]


def test_quadratic_certificates_cover_interiors_and_endpoints(certificate):
    for row in certificate["rows"].values():
        for name, point, limit in (
            ("poly", "vertex", row["hmax"]),
            ("neg_poly", "neg_vertex", certificate["shift"]),
            ("cap_poly", "cap_vertex", row["hmax"]),
        ):
            c0, c1, c2 = row[name]
            x = row[point]
            value = c0 + c1 * x + c2 * x * x
            assert c2 < 0 and 0 <= x <= limit
            assert value >= c0 and value >= c0 + c1 * limit + c2 * limit * limit
            if 0 < x < limit:
                assert c1 + 2 * c2 * x == 0 and value == c0 - c2 * x * x
        assert row["pos_out"] < Q(91, 1000)
        assert row["pos_in"] < Q(89, 1000)
        assert row["neg_in"] < Q(19, 1000)
        assert row["neg_out"] < Q(70, 1000)


def test_independent_diameter_only_envelopes_also_exclude(certificate):
    # A separate weaker bound uses t<=D rather than the hull-escape t<z+s.
    c = certificate
    k, b, w = c["law"]["pressure"]
    f = c["law"]["factors"]["OZ_dnfr_factor"]
    maximum = c["law"]["capacity_rails"][1]
    cut = c["law"]["thresholds"]["dnfr_hi"]
    inward_capacity = maximum * (c["r0"] + c["r1"] * c["shift"])
    inward_denominator = 1 - c["lam"] / 2 - DT * k * f * inward_capacity
    for row in c["rows"].values():
        kt = c["lam"] * row["ca"] / 2 + row["ct"]
        sz = c["lam"] * row["ct"] / 2 + row["ch"]
        n = c["lam"] * k + w / PI * (
            (c["lam"] * row["ca"] / 2 + row["ct"]) * row["diameter"]
            + (c["lam"] * row["ct"] / 2 + row["ch"]) * row["hmax"]
        )
        u = row["negative_capacity"]
        upper = f * u * n / (1 - c["lam"] / 2 - DT * k * f * u)
        assert 0 < upper < Q(78, 1000) < c["xi"]
        # Here h>=0, so the -S*h contribution may be discarded upward.
        negative_inward_n = c["lam"] * k + w / PI * kt * row["diameter"]
        negative_inward = f * inward_capacity * negative_inward_n / inward_denominator
        assert 0 < negative_inward < Q(30, 1000) < c["xi"]
        positive_inward_n = (
            cut * (2 * k + b * maximum / 2)
            + c["lam"] * k
            + w / PI * ((cut + kt) * row["diameter"] + sz * c["shift"])
        )
        positive_inward = f * inward_capacity * positive_inward_n / inward_denominator
        assert 0 < positive_inward < Q(86, 1000) < c["xi"]


def test_oz_fallback_zero_peak_and_signed_zero_remain_covered(certificate):
    tiny, fallback = Q(1e-9), Q(0.1)
    for p in (-tiny, Q(0), tiny):
        assert abs(p) <= tiny
        for row in certificate["rows"].values():
            assert fallback * row["capacity_max"] == row["fallback"]
            assert row["fallback"] < Q(76, 1000) < certificate["xi"]
    # P'=0 gives p0'=0,rho0=0. Multiplying the selection inequality by P'
    # is safe; both directions cover the real zero's missing IEEE sign bit.
    assert {direction for direction, _, _ in CELLS} == {-1, 1}
    assert all(
        max(
            row[name] for name in ("pos_out", "pos_in", "neg_out", "neg_in", "fallback")
        )
        < certificate["xi"]
        for row in certificate["rows"].values()
    )
