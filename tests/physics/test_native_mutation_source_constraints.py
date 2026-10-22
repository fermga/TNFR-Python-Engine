"""Exact source constraints; negative-outward joint feasibility stays OPEN.

Freeze the unchanged paired unit doubled-C5/default-law/two-call sector of
the orientation owner. First actual OZ0/high-Si ILrest must acquire a positive
signed secant; second actual ZHIR0/high-Si ILrest must complete renewal.
Only negative-outward target-minimum geometry remains. Retain actual maximum
capacity ownership, both fresh pressure fields, both IL masks, common charts,
the common form offset across all fifteen source/flow coordinates, and the
actual carried controller branches. No native source is evaluated here.

The prospective budget is nine gain pairs times near/far maximum ownership.
Stage1 evaluates one coarse ownership envelope. Stage2 evaluates exactly one
coupled envelope from the actual first OZ pressure ratio and the second near
IL masks. Rational Bernstein controls use only natural phase breakpoints;
there is no grid, optimizer, LP, native trajectory or adaptive refinement.
Both stages are inconclusive. Positive denominator admission is proved, but
their rate upper bounds exceed xi. This neither constructs a source nor proves
that the true joint constraints are feasible. No failed bound is overwritten.

The source inverse is a declared-law inverse, not pressure reconstructed from
a response. Basis/tensor and scalar mask controls below are algebraic controls,
not prepared physical sources. Ideal common lifts and represented non-angle
coefficients do not establish native binary64 wrap or signed-zero realization.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from math import comb

import pytest

from tests.physics import test_native_mutation_joint_orientation as prior

configuration = prior.configuration
no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard
DT, PI, BRANCHES = prior.DT, prior.PI, prior.BRANCHES
CELLS = tuple(product(BRANCHES, BRANCHES, ("near", "far")))
DOMAIN = (
    "unchanged paired C5/defaults/held Euler dt1/16/two calls/Gamma0",
    "OZ0+high-Si ILrest; acquired ZHIR0+high-Si ILrest; complete renewal",
    "negative shift; h=-z<0; target renewed minimum; pnext0<=0",
    "actual max owner near/far; source and endpoint laws retained jointly",
    "one source form offset across all15 source/flow coordinates",
    "18 ownership/gain cells; one coarse then one coupled envelope",
    "exact quadratic/Bernstein controls; only natural phase breakpoints",
    "no sources/optimizer/LP/native execution; joint feasibility OPEN",
)


def _law(configuration):
    return prior.prior.geometry.prior.previous.enclosure._law(configuration)


def _linear_value(poly, value):
    return sum(coefficient * value**i for i, coefficient in enumerate(poly))


def _rate_upper(xi, gap_controls, den_controls):
    """Bound xi-gap/den with the sign-sensitive positive divisor endpoint."""
    gap = min(gap_controls)
    denominator = min(den_controls) if gap < 0 else max(den_controls)
    assert min(den_controls) > 0
    return xi - gap / denominator


@pytest.fixture(scope="module")
def certificate(configuration, renewal_guard):
    law = _law(configuration)
    k, b, w = law["pressure"]
    alpha, beta, gamma = law["stability"]
    assert (alpha, beta, gamma) == (w, k, b)
    assert law["form_rails"] == (Q(-1), Q(1)) and law["tau"] > 2
    assert min(law["lag_limits"]) >= 2
    assert configuration.graph["OZ_NOISE_MODE"] is False
    maximum, d = law["capacity_rails"][1], w / PI
    factor, retain = (
        law["factors"][key] for key in ("OZ_dnfr_factor", "IL_dnfr_factor")
    )
    cut, low, high = (law["thresholds"][key] for key in ("dnfr_hi", "si_lo", "si_hi"))
    xi = Q(configuration.graph["ZHIR_THRESHOLD_XI"])
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    floor = (high - beta - gamma) / alpha
    r0, r1 = (low - beta - gamma * (1 - cut)) / alpha, beta / (alpha * PI)
    zeta, lam = DT * k * retain, DT * k * retain * maximum
    assert Q(1, 3) < floor < 1 and r0 > 0
    assert 1 - lam / 2 - DT * k * factor * maximum > 0
    rows = {}
    for first, second in product(BRANCHES, repeat=2):
        update = prior.prior.geometry.prior._update_gains
        g1, local1 = update(law, law["phase_gains"], first)
        g2, local2 = update(law, (g1, local1), second)
        a1, a2 = 1 - g1 - local1, 1 - g2 - local2
        assert 3 * (1 - g1 - local1 / 2) + shift < 3 < PI
        det = a1 * (a1 - local1 / 2) - local1 * local1 / 4
        ca, ct, ch = (a1 - local1 / 2) / det, local1 / (2 * det), a1 / det - 1
        ephase, fphase = ca + ct / 2, ct + (1 + ch) / 2
        diameter = a2 * shift / (g2 + local2 / 2)
        height = min(a2 * shift / (g2 + local2), 2 * k / d)
        assert 0 < shift < diameter / 2 < height and ch > ct > 0
        assert r0 + r1 * height < floor  # Target cannot own the maximum.

        def capacity(z):
            return maximum * (r0 + r1 * z)

        def denominator(u):
            return 1 - lam / 2 - DT * k * factor * u

        kt, sz = lam * ca / 2 + ct, lam * ct / 2 + ch
        qnear, qfar = lam * (1 - floor) * ephase / 2, lam * (1 - floor) * fphase / 2
        assert kt > qfar
        contrast_ratio = (1 - floor) / (1 + floor)
        assert ct + zeta * maximum * min(ca / 2 - contrast_ratio * fphase, Q(0)) > 0
        extra = max(qnear - kt * g2, Q(0))
        cap_poly = (2 * k * r0, 2 * k * r1 - d * r0, -d * r1)
        cap_max, cap_point = prior._maximum(cap_poly, Q(0), height)
        near_cap = cap_max / (b * ((1 + floor) / 2 - r0 - r1 * height))
        near_cap = min(capacity(height), near_cap)
        assert denominator(near_cap) > 0
        coarse_controls, near_candidates, far_candidates = [], [], []
        for owner, resource in (("near", (1 - floor) / 2), ("far", (1 - floor) / 4)):
            constant = lam * (k + b * maximum * resource)
            pieces = (
                (
                    constant + d * (kt * a2 * shift + extra * diameter),
                    d * (kt * a2 + sz - extra),
                    Q(0),
                    diameter / 2,
                ),
                (
                    constant + d * (kt * a2 * shift + extra * a2 * shift / g2),
                    d * (kt * a2 + sz - extra * (g2 + local2) / g2),
                    diameter / 2,
                    height,
                ),
            )
            for n0, n1, left, right in pieces:
                if owner == "near":
                    value = max(n0 + n1 * left, n0 + n1 * right)
                    near_candidates.append(
                        factor * near_cap * value / denominator(near_cap)
                    )
                else:
                    poly = (r0 * n0, r1 * n0 + r0 * n1, r1 * n1)
                    value, _ = prior._maximum(poly, left, right)
                    far_candidates.append(
                        factor * maximum * value / denominator(capacity(height))
                    )
                coarse_controls.append((owner, n0, n1, left, right))

        # Stage2 is frozen as one common relaxation of both ownership cases.
        # Retain the ACTUAL t in the near IL mask. For t<-z, mapping t to -z
        # raises the transport numerator and weakens its absolute phase cost.
        radius, rail = (r0, r1), (2 * k / b, -d / b)
        product_rb = prior._multiply(radius, rail)
        intercept = prior._add((Q(1),), prior._scale(rail, -zeta / cut))
        assert min(prior._bernstein(intercept, Q(0), height)) > 0
        intervals = ((Q(0), diameter / 2), (diameter / 2, height))
        coupled_controls, coupled_bounds = [], []
        for endpoint, tpoly in (("lower", (Q(0), Q(-1))), ("upper", (a2 * shift, a2))):
            mask = prior._add(
                (floor,),
                prior._scale(prior._add((Q(0), Q(1)), tpoly), r1 / 2),
                prior._scale(radius, Q(-1)),
            )
            den = prior._add(
                prior._multiply(mask, intercept),
                prior._scale(product_rb, -(DT * k * factor + zeta / cut)),
            )
            phase = prior._add((Q(0), ch), prior._scale(tpoly, ct))
            num = prior._scale(prior._multiply(product_rb, phase), factor * d)
            gap = prior._add(prior._scale(den, xi), prior._scale(num, Q(-1)))
            for left, right in intervals:
                den_controls = prior._bernstein(den, left, right)
                gap_controls = prior._bernstein(gap, left, right)
                assert min(den_controls) > 0
                upper = _rate_upper(xi, gap_controls, den_controls)
                coupled_bounds.append(upper)
                coupled_controls.append(
                    (
                        endpoint,
                        tpoly,
                        den,
                        num,
                        gap,
                        left,
                        right,
                        den_controls,
                        gap_controls,
                    )
                )
        coarse_near, coarse_far, coupled = (
            max(near_candidates),
            max(far_candidates),
            max(coupled_bounds),
        )
        assert coupled > xi and coarse_far > xi
        rows[(first, second)] = dict(
            gains1=(g1, local1),
            gains2=(g2, local2),
            ca=ca,
            ct=ct,
            ch=ch,
            diameter=diameter,
            height=height,
            near_cap=near_cap,
            cap_poly=cap_poly,
            cap_point=cap_point,
            cap_max=cap_max,
            coarse=tuple(coarse_controls),
            coarse_near=coarse_near,
            coarse_far=coarse_far,
            coupled=coupled,
            controls=tuple(coupled_controls),
        )
    digest = sha256(repr((DOMAIN, CELLS, law, rows)).encode()).hexdigest()
    return dict(law=law, rows=rows, xi=xi, floor=floor, digest=digest)


def _laplacian(values):
    return tuple(
        (values[(i - 1) % 5] + values[(i + 1) % 5]) / 2 - values[i] for i in range(5)
    )


def _inverse_pressure(pressure):
    return tuple(
        (-4 * pressure[i] + 2 * pressure[(i - 2) % 5] + 2 * pressure[(i + 2) % 5]) / 5
        for i in range(5)
    )


def _offset_interval(boundaries):
    flattened = tuple(value for boundary in boundaries for value in boundary)
    return -1 - min(flattened), 1 - max(flattened)


def test_guarded_protocol_preserves_inconclusive_bounds(certificate, renewal_guard):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert len(CELLS) == 18 and len(certificate["rows"]) == 9
    assert len(certificate["digest"]) == 64
    rows = tuple(certificate["rows"].values())
    assert Q(109, 1000) < max(row["coarse_near"] for row in rows) < Q(110, 1000)
    assert Q(174, 1000) < max(row["coarse_far"] for row in rows) < Q(175, 1000)
    assert Q(169, 1000) < max(row["coupled"] for row in rows) < Q(170, 1000)
    assert any(row["coarse_near"] < certificate["xi"] for row in rows)
    assert all(row["coupled"] > certificate["xi"] for row in rows)


def test_actual_capacity_owner_corner_bounds(certificate):
    floor = certificate["floor"]
    for left, right, other in product((floor, Q(1)), repeat=3):
        near, contrast = (left + right) / 2, (left - right) / 2
        # The far node2 owns V; node3 has normalized capacity other.
        resource = near * (near - (1 + other) / 2) / 2
        resource += contrast * contrast - contrast * (1 - other) / 4
        assert resource <= (1 - floor) / 4
    # At fixed other, each near coordinate has positive second derivative;
    # increasing the other far capacity decreases the resource. Thus the
    # displayed corners cover the full cube and retain actual V ownership.
    assert (1 - floor) / 4 > 3 * (1 - floor) ** 2 / 8
    assert (1 + floor) / 2 > floor


def test_source_oz_peak_transport_keeps_its_strict_slack(certificate):
    law = certificate["law"]
    cut = law["thresholds"]["dnfr_hi"]
    # Algebraic slack coordinates, not a claimed pressure field or source:
    # P=p0/c-epsilon, p_near=-P+s_near. The strict positive contribution
    # n*epsilon must survive averaging capacities and pressure separately.
    for left, right, slack1, slack4 in product((Q(1, 3), Q(1)), repeat=4):
        target, epsilon = Q(1), Q(1, 7)
        peak = target / cut - epsilon
        weighted = (left * (-peak + slack1) + right * (-peak + slack4)) / 2
        near = (left + right) / 2
        assert (
            weighted + near * target / cut
            == (near * epsilon + (left * slack1 + right * slack4) / 2)
            > 0
        )
    # Zero peak needs its unhomogenized Si mask; multiplying by P=0 erases
    # genuine rejection. This scalar boundary is not a native preparation.
    alpha, beta, gamma = law["stability"]
    assert beta + gamma < law["thresholds"]["si_hi"]
    assert Q(0) * (alpha * Q(0) + beta + gamma - law["thresholds"]["si_hi"]) == 0


def test_declared_pressure_inverse_and_two_flow_rows(certificate):
    law = certificate["law"]
    k, b, w = law["pressure"]
    d = w / PI
    factor, retain = law["factors"]["OZ_dnfr_factor"], law["factors"]["IL_dnfr_factor"]
    # LQ=I-J/5 on all coordinate bases, not merely an observed response.
    for j in range(5):
        basis = tuple(Q(i == j) for i in range(5))
        centered = tuple(x - Q(1, 5) for x in basis)
        assert _laplacian(_inverse_pressure(basis)) == centered
        assert _inverse_pressure(_laplacian(basis)) == centered
        assert _laplacian(_inverse_pressure(_laplacian(basis))) == _laplacian(basis)
        assert _inverse_pressure(
            _laplacian(_inverse_pressure(basis))
        ) == _inverse_pressure(basis)
    first = certificate["rows"][("stable", "stable")]["gains1"]
    matrix = prior.prior.geometry.prior._phase_matrix(*first)
    unweighted_changes = []
    for pressure_index, capacity_index, phase_index in product(
        range(4), range(5), range(5)
    ):
        pressure = tuple(Q(i == pressure_index) - Q(i == 4) for i in range(5))
        # The positive basis 1+e_j is independent (I+J is invertible), and
        # permits the reciprocal-capacity balances without a zero divisor.
        capacity = tuple(1 + Q(i == capacity_index) for i in range(5))
        theta = tuple(Q(i == phase_index) for i in range(5))
        inverse = _inverse_pressure(pressure)
        x0 = tuple((inverse[i] - b * capacity[i] - d * theta[i]) / k for i in range(5))
        assert (
            tuple(
                k * _laplacian(x0)[i]
                + b * _laplacian(capacity)[i]
                + d * _laplacian(theta)[i]
                for i in range(5)
            )
            == pressure
        )
        flow1 = tuple(
            capacity[i] * (factor if i == 0 else retain) * pressure[i] for i in range(5)
        )
        x1 = tuple(x0[i] + DT * flow1[i] for i in range(5))
        phase1 = prior.prior.geometry.prior._mv(matrix, theta)
        predicted = tuple(
            pressure[i]
            + DT * k * _laplacian(flow1)[i]
            + d * _laplacian(tuple(phase1[j] - theta[j] for j in range(5)))[i]
            for i in range(5)
        )
        direct = tuple(
            k * _laplacian(x1)[i]
            + b * _laplacian(capacity)[i]
            + d * _laplacian(phase1)[i]
            for i in range(5)
        )
        assert direct == predicted
        assert sum(direct) == 0
        flow2 = tuple(
            capacity[i] * (1 if i == 0 else retain) * predicted[i] for i in range(5)
        )
        x2 = tuple(x1[i] + DT * flow2[i] for i in range(5))
        assert tuple((x2[i] - x1[i]) / DT for i in range(5)) == flow2
        assert (
            sum(
                (x1[i] - x0[i]) / (capacity[i] * (factor if i == 0 else retain))
                for i in range(5)
            )
            == 0
        )
        assert (
            sum(
                (x2[i] - x1[i]) / (capacity[i] * (1 if i == 0 else retain))
                for i in range(5)
            )
            == 0
        )
        unweighted_changes.append(sum(x1[i] - x0[i] for i in range(5)))
    assert any(value != 0 for value in unweighted_changes)


def test_one_form_offset_is_shared_across_all_fifteen_rails():
    boundaries = (
        (Q(-1), Q(0), Q(0), Q(0), Q(0)),
        (Q(0), Q(1), Q(0), Q(0), Q(0)),
        (Q(0), Q(0), Q(2), Q(0), Q(0)),
    )
    # Each time separately fits in the rails after some translation; one
    # physical source must use the SAME translation at all three times.
    assert all(
        _offset_interval((boundary,))[0] <= _offset_interval((boundary,))[1]
        for boundary in boundaries
    )
    assert _offset_interval(boundaries)[0] > _offset_interval(boundaries)[1]
    admissible = tuple(
        tuple(value / 2 for value in boundary) for boundary in boundaries
    )
    lower, upper = _offset_interval(admissible)
    assert lower <= upper
    flattened = tuple(value for boundary in admissible for value in boundary)
    assert upper - lower == 2 - (max(flattened) - min(flattened))
    for shift in (lower, (lower + upper) / 2, upper):
        assert all(
            -1 <= value + shift <= 1 for boundary in admissible for value in boundary
        )


def test_coupled_denominator_and_bernstein_identity(certificate):
    # A negative gap lower bound requires the SMALL divisor. Using D_max
    # would miss xi-G/D at G=-1,D=1, despite valid G and D enclosures.
    xi = certificate["xi"]
    for gaps in ((Q(-1), Q(1)), (Q(1), Q(2)), (Q(0), Q(1))):
        divisors = (Q(1), Q(2))
        upper = _rate_upper(xi, gaps, divisors)
        assert all(xi - gap / den <= upper for gap, den in product(gaps, divisors))
        assert any(xi - gap / den == upper for gap, den in product(gaps, divisors))
    assert xi - Q(-1) / 1 > xi - Q(-1) / 2
    for row in certificate["rows"].values():
        for _, _, den, num, gap, left, right, den_controls, gap_controls in row[
            "controls"
        ]:
            assert min(den_controls) > 0
            for t in (Q(0), Q(1, 3), Q(2, 3), Q(1)):
                z = left + (right - left) * t
                for poly, controls in ((den, den_controls), (gap, gap_controls)):
                    observed = sum(
                        controls[j] * comb(3, j) * t**j * (1 - t) ** (3 - j)
                        for j in range(4)
                    )
                    assert observed == _linear_value(poly, z)
                    assert min(controls) <= observed <= max(controls)
                assert _linear_value(gap, z) == (
                    certificate["xi"] * _linear_value(den, z) - _linear_value(num, z)
                )
                assert _linear_value(num, z) / _linear_value(den, z) <= _rate_upper(
                    xi, gap_controls, den_controls
                )
        assert min(min(control[-1]) for control in row["controls"]) < 0


def test_actual_t_elimination_is_linear_fractional(certificate):
    law = certificate["law"]
    k, _, w = law["pressure"]
    factor, retain = law["factors"]["OZ_dnfr_factor"], law["factors"]["IL_dnfr_factor"]
    alpha, beta, gamma = law["stability"]
    cut, low = law["thresholds"]["dnfr_hi"], law["thresholds"]["si_lo"]
    b = law["pressure"][1]
    r0, r1 = (low - beta - gamma * (1 - cut)) / alpha, beta / (alpha * PI)
    d, zeta = w / PI, DT * k * retain
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    for row in certificate["rows"].values():
        g2, local2 = row["gains2"]
        a2 = 1 - g2 - local2
        for z in (Q(0), row["height"] / 2, row["height"]):
            radius, rail = r0 + r1 * z, (2 * k - d * z) / b
            lower, upper = -z, a2 * (shift + z)

            def values(t):
                mask = certificate["floor"] + r1 * (z + t) / 2 - radius
                u = radius * rail / mask
                phase = row["ch"] * z + row["ct"] * t
                numerator = factor * d * radius * rail * phase
                denominator = mask * (1 - zeta * rail / cut)
                denominator -= (DT * k * factor + zeta / cut) * radius * rail
                assert mask > 0 and denominator > 0
                original = (
                    factor
                    * d
                    * u
                    * phase
                    / (1 - DT * k * factor * u - zeta * (u + rail) / cut)
                )
                assert original == numerator / denominator
                return numerator, denominator

            n0, d0 = values(lower)
            n1, d1 = values(upper)
            for mix in (Q(1, 3), Q(2, 3)):
                numerator, denominator = values((1 - mix) * lower + mix * upper)
                assert numerator == (1 - mix) * n0 + mix * n1
                assert denominator == (1 - mix) * d0 + mix * d1
                assert (
                    min(n0 / d0, n1 / d1)
                    <= numerator / denominator
                    <= max(n0 / d0, n1 / d1)
                )
