"""Same-source necessary orientation for acquired Mutation renewal.

Freeze the previous heterogeneous doubled-C5/default-law/two-call domain:
paired phases/capacities/forms, unit support, source phase diameter <=3,
all three form boundaries in [-1,1], held Euler dt1/16, zero Gamma and
supplied AL/IL/EN context with prior IL, ordinal3 and otherwise zero memory.
Require actual OZ0/high-Si ILrest then acquired ZHIR0/high-Si ILrest, with
both pressure fields from the SAME source and complete phase renewal.

The frozen budget has 18 gain/direction cells, signed-row and capacity-cube
extrema and exact piecewise quadratics. A declared extension permits cubic
Bernstein bounds only on natural breakpoints 0,T/2,H,nbar=VF_MAX. No source,
optimizer, LP, native trajectory or old response fixture is evaluated.

Same-source form/capacity/phase correlations exclude every positive shift,
negative inward action and adjacent renewed extreme. Success would require
a negative shift, h<0 and the target as renewed minimum. Second pressure
is <=0, including negative signed zero; strict reversal is not proved.
The frozen negative-outward relaxation reaches .157164>xi: it is unresolved,
not a feasible source or a counterexample. Joint realizability stays OPEN.
These are ideal common-lift proofs, not binary64 wrap-equivariance claims.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from math import comb

import pytest

from tests.physics import test_native_mutation_capacity_budget as prior

configuration = prior.configuration
no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard
DT, PI, BRANCHES = prior.DT, prior.PI, prior.BRANCHES
CELLS = tuple(product((-1, 1), BRANCHES, BRANCHES))
# Coefficients of the four actual non-target L-phase rows on (t,h,r,s).
GRADIENTS = (
    (Q(-1, 2), Q(1, 2), Q(-1), Q(1, 2)),
    (Q(1, 2), Q(0), Q(1, 2), Q(-3, 2)),
    (Q(1, 2), Q(0), Q(-1, 2), Q(3, 2)),
    (Q(-1, 2), Q(1, 2), Q(1), Q(-1, 2)),
)
DOMAIN = (
    "same-source arbitrary paired C5 forms/phases/capacities; actual maximum V",
    "source and both flow forms in rails; source phase diameter<=3",
    "OZ0+high-Si ILrest then acquired ZHIR0+high-Si ILrest; full renewal",
    "AL/IL/EN/priorIL/ordinal3; empty t1 history; zero remaining memory",
    "defaults/strict grammar/Gamma0/held Euler dt1/16/two calls",
    "18 gain/direction cells; source row/cube support; piecewise quadratics",
    "degree<=3 Bernstein; only 0,T/2,H,nbar=VF_MAX; no adaptive refinement",
    "no sources/optimizer/LP/native responses; negative-outward unresolved",
)


def _maximum(poly, lower, upper):
    points = [lower, upper]
    if poly[2] < 0:
        vertex = -poly[1] / (2 * poly[2])
        if lower <= vertex <= upper:
            points.append(vertex)
    return max((sum(c * x**j for j, c in enumerate(poly)), x) for x in points)


def _add(*polys):
    result = [Q(0)] * max(map(len, polys))
    for poly in polys:
        for i, value in enumerate(poly):
            result[i] += value
    return tuple(result)


def _scale(poly, value):
    return tuple(value * x for x in poly)


def _multiply(left, right):
    result = [Q(0)] * (len(left) + len(right) - 1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[i + j] += x * y
    return tuple(result)


def _bernstein(poly, lower, upper):
    """Degree-three rational Bernstein enclosure, including lower degrees."""
    assert len(poly) <= 4 and lower < upper
    local = tuple(
        sum(
            poly[j] * comb(j, i) * lower ** (j - i) * (upper - lower) ** i
            for j in range(i, len(poly))
        )
        for i in range(4)
    )
    return tuple(
        sum(local[j] * Q(comb(i, j), comb(3, j)) for j in range(i + 1))
        for i in range(4)
    )


@pytest.fixture(scope="module")
def certificate(configuration, renewal_guard):
    # Shared configuration only: no prior certificate or evaluated source.
    law = prior.geometry.prior.previous.enclosure._law(configuration)
    k, b, w = law["pressure"]
    alpha, beta, gamma = law["stability"]
    assert (alpha, beta, gamma) == (w, k, b)
    assert law["form_rails"] == (Q(-1), Q(1)) and law["tau"] > 2
    assert min(law["lag_limits"]) >= 2
    assert configuration.graph["OZ_NOISE_MODE"] is False
    vmax, d = law["capacity_rails"][1], w / PI
    retain, factor = (
        law["factors"][key] for key in ("IL_dnfr_factor", "OZ_dnfr_factor")
    )
    cut, low, high = (law["thresholds"][key] for key in ("dnfr_hi", "si_lo", "si_hi"))
    xi = Q(configuration.graph["ZHIR_THRESHOLD_XI"])
    shift = law["factors"]["ZHIR_theta_shift_factor"] * PI / 4
    floor = (high - beta - gamma) / alpha
    r0 = (low - beta - gamma * (1 - cut)) / alpha
    r1 = beta / (alpha * PI)
    zeta, lam = DT * k * retain, DT * k * retain * vmax
    assert Q(1, 3) < floor < 1 and r0 > 0
    assert 1 - lam / 2 - DT * k * factor * vmax > 0
    # Exact coordinatewise-convex near-capacity extrema; far capacities
    # minimize at floor*V. These are coefficient corners, not sources.
    profiles = (
        # The V=0 endpoint is an algebraic closure, not an active source.
        # The complete numerator is convex in V after the near-capacity
        # corner bound; include both endpoints rather than assume that its
        # signed phase support is monotone in capacity.
        (Q(0), Q(0), Q(0)),
        (floor, Q(0), Q(0)),
        ((1 + floor) / 2, (1 - floor) / 2, (1 - floor) * (3 - floor) / 8),
        ((1 + floor) / 2, -(1 - floor) / 2, (1 - floor) * (3 - floor) / 8),
        (Q(1), Q(0), (1 - floor) / 2),
    )
    rows = {}
    for first, second in product(BRANCHES, repeat=2):
        g1, l1 = prior.geometry.prior._update_gains(law, law["phase_gains"], first)
        g2, l2 = prior.geometry.prior._update_gains(law, (g1, l1), second)
        a1, a2 = 1 - g1 - l1, 1 - g2 - l2
        assert 3 * (1 - g1 - l1 / 2) + shift < 3 < PI
        assert a1 > 2 * l1 and a2 >= l2 / 2
        det = a1 * (a1 - l1 / 2) - l1 * l1 / 4
        ca, ct, ch = (a1 - l1 / 2) / det, l1 / (2 * det), a1 / det - 1
        ephase, fphase = ca + ct / 2, ct + (1 + ch) / 2
        diameter, height = a2 * shift / (g2 + l2 / 2), a2 * shift / (g2 + l2)

        def capacity(h):
            return vmax * (r0 + r1 * h)

        def denominator(u):
            return 1 - lam / 2 - DT * k * factor * u

        assert 0 < shift < diameter / 2 < height < diameter
        assert det > 0 and fphase < ca and capacity(height) < Q(752, 1000)
        assert r0 + r1 * height < floor
        peak_constant = cut * (2 * k + b * vmax * (1 - floor / 2))
        controls, positive_bounds = [], []
        for near, contrast, resource in profiles:
            kt = lam * near * ca / 2 + ct
            # Drop the nonpositive -lambda_n*ct*h/2 for positive outward
            # h>=0. The two V endpoints above cover remaining signed terms
            # by convexity; no monotonic phase-support claim is needed.
            sz = ch
            qr, qs = lam * contrast * ephase, lam * contrast * fphase
            constant = (peak_constant if near else cut * 2 * k) + lam * (
                k * near + b * vmax * resource
            )
            for gradient, sign in product(GRADIENTS, (-1, 1)):
                gt, gh, gr, gs = gradient
                aa, bb = kt + cut * sign * gt, -sz + cut * sign * gh
                cc, ee = qr + cut * sign * gr, -qs + cut * sign * gs
                assert aa >= 0
                remainder = max(abs(ee) - aa, Q(0)) / 2
                lifted, absolute = max(aa, abs(cc)), abs(cc)
                pieces = (
                    (
                        constant + d * (lifted + remainder) * diameter,
                        d * (bb - lifted + absolute),
                        Q(0),
                        diameter / 2,
                    ),
                    (
                        constant
                        + d * (lifted + remainder + absolute) * a2 * shift / g2,
                        d
                        * (
                            (bb + remainder)
                            - (lifted + remainder + absolute) * (g2 + l2) / g2
                        ),
                        diameter / 2,
                        height,
                    ),
                )
                for n0, n1, lower, upper in pieces:
                    poly = (r0 * n0, r1 * n0 + r0 * n1, r1 * n1)
                    value, point = _maximum(poly, lower, upper)
                    positive_bounds.append(
                        factor * vmax * value / denominator(capacity(height))
                    )
                    controls.append(
                        (poly, lower, upper, point, value, aa, ee, remainder)
                    )
        kt, sz = lam * ca / 2 + ct, lam * ct / 2 + ch
        qnear, qfar = lam * (1 - floor) * ephase / 2, lam * (1 - floor) * fphase / 2
        assert kt > max(qnear, qfar)
        constant = peak_constant + lam * (k + b * vmax * (1 - floor) / 2)
        inward_u = capacity(shift)
        positive_in = (
            factor
            * inward_u
            * (constant + d * ((cut + kt) * diameter + sz * shift))
            / denominator(inward_u)
        )
        negative_in = (
            factor
            * inward_u
            * (constant - peak_constant + d * kt * diameter)
            / denominator(inward_u)
        )
        # A negative inward action requires the target less than shift above
        # the old minimum. Thus the near phase mean is <shift; qnear<kt and
        # 2*shift<diameter justify the coarser kt*diameter numerator above.
        adjacent = (
            factor
            * inward_u
            * (constant + d * (cut + kt + sz + (qnear + qfar) / 2) * shift)
            / denominator(inward_u)
        )
        fallback = Q(0.1) * capacity(height)

        # Frozen unsuccessful refinement: no inference of source existence.
        # Negative fresh target pressure yields n<=u+(2k-d*z)/b.
        near_line = (vmax * r0 + 2 * k / b, vmax * r1 - d / b)
        assert near_line[1] < 0
        kink = (vmax - near_line[0]) / near_line[1]
        breaks = sorted(
            {Q(0), diameter / 2, height} | ({kink} if 0 < kink < height else set())
        )
        bernstein_controls, negative_bounds = [], []
        ratio = (1 - floor) / (1 + floor)
        assert ct + zeta * vmax * min(ca / 2 - ratio * fphase, Q(0)) > 0
        for left, right in zip(breaks, breaks[1:]):
            mid = (left + right) / 2
            near = (vmax,) if near_line[0] + near_line[1] * mid >= vmax else near_line
            available = (
                (diameter, Q(-1))
                if mid < diameter / 2
                else (a2 * shift / g2, -(g2 + l2) / g2)
            )
            kk = _add((ct,), _scale(near, zeta * ca / 2))
            ss = _add((ch,), _scale(near, zeta * ct / 2))
            extra = _scale(near, zeta * max(ratio * ephase - ca * g2 / 2, Q(0)))
            phase = _add(
                _multiply(kk, (a2 * shift, a2)),
                _multiply(ss, (Q(0), Q(1))),
                _multiply(extra, available),
            )
            numerator = _add(
                _scale(near, zeta * (k + b * vmax * (1 - floor) / 2)), _scale(phase, d)
            )
            poly = _scale(_multiply((r0, r1), numerator), factor * vmax)
            coefficients = _bernstein(poly, left, right)
            negative_bounds.append(max(coefficients) / denominator(capacity(right)))
            bernstein_controls.append((poly, left, right, coefficients))

        positive_out, unresolved = max(positive_bounds), max(negative_bounds)
        assert positive_out < Q(95, 1000) < xi
        assert positive_in < Q(94, 1000) < xi
        assert negative_in < Q(32, 1000) < xi
        assert adjacent < Q(53, 1000) < xi
        assert fallback < Q(76, 1000) < xi
        assert unresolved > xi
        rows[(first, second)] = dict(
            gains1=(g1, l1),
            gains2=(g2, l2),
            ca=ca,
            ct=ct,
            ch=ch,
            diameter=diameter,
            height=height,
            controls=tuple(controls),
            positive_out=positive_out,
            positive_in=positive_in,
            negative_in=negative_in,
            adjacent=adjacent,
            fallback=fallback,
            unresolved=unresolved,
            bernstein=tuple(bernstein_controls),
        )
    digest = sha256(
        repr((DOMAIN, CELLS, GRADIENTS, law, profiles, rows)).encode()
    ).hexdigest()
    return dict(
        law=law, rows=rows, floor=floor, profiles=profiles, xi=xi, digest=digest
    )


def test_guarded_scope_is_an_orientation_obstruction(certificate, renewal_guard):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    assert len(CELLS) == 18 and len(certificate["rows"]) == 9
    assert len(certificate["digest"]) == 64
    maximum = max(row["unresolved"] for row in certificate["rows"].values())
    assert Q(157, 1000) < maximum < Q(158, 1000)
    # An upper envelope above xi neither supplies history nor proves success.
    assert all(
        row["unresolved"] > certificate["xi"] for row in certificate["rows"].values()
    )


def test_same_source_sink_has_fixed_form_signs_and_capacity_corners(certificate):
    floor = certificate["floor"]
    laplacian = prior.geometry.prior._laplacian
    for left, right in product((floor, Q(1)), repeat=2):
        near, contrast = (left + right) / 2, (left - right) / 2
        coefficients = (
            (near + 2 * contrast) / 4,
            (near - 2 * contrast) / 4,
            -(near + contrast) / 4,
            -(near - contrast) / 4,
        )
        assert 2 * abs(contrast) < near
        assert sum(abs(x) for x in coefficients) == near
        for i in range(5):
            x = tuple(Q(j == i) for j in range(5))
            pressure = laplacian(x)
            observed = near * (pressure[2] + pressure[3]) / 2
            observed -= contrast * (pressure[1] - pressure[4]) / 2
            expected = sum(c * x[j] for c, j in zip(coefficients, (1, 4, 2, 3)))
            assert observed == expected
        resource = near * (near - floor) / 2 + contrast * contrast
        assert resource <= (1 - floor) / 2
        assert resource <= near * (1 - floor) / 2

    # Each near coordinate has positive second derivative 3/4; each far
    # coordinate has negative coefficient -(near +/- contrast)/4. Thus these
    # exact corners bound the full box, not merely the four checked points.
    def cap(a, b):
        return 3 * (a * a + b * b) / 8 - a * b / 4 - floor * (a + b) / 4

    for other in (floor, Q(1)):
        assert cap(2, other) - 2 * cap(1, other) + cap(0, other) == Q(3, 4)


def test_phase_inverse_and_actual_signed_peak_rows(certificate):
    laplacian = prior.geometry.prior._laplacian
    for row in certificate["rows"].values():
        g1, l1 = row["gains1"]
        matrix = prior.geometry.prior._phase_matrix(g1, l1)
        for i in range(5):
            old = tuple(Q(j == i) for j in range(5))
            new = prior.geometry.prior._mv(matrix, old)
            near, far = (new[1] + new[4]) / 2, (new[2] + new[3]) / 2
            t, h, r, s = (
                near - far,
                new[0] - near,
                (new[1] - new[4]) / 2,
                (new[2] - new[3]) / 2,
            )
            ca, ct, ch = row["ca"], row["ct"], row["ch"]
            old_near, old_far = (old[1] + old[4]) / 2, (old[2] + old[3]) / 2
            assert old_near - old_far == ca * t - ct * h
            assert old[0] - old_near - h == ch * h - ct * t
            assert (old[1] - old[4]) / 2 - (old[2] - old[3]) / 4 == (
                (ca + ct / 2) * r - (ct + (1 + ch) / 2) * s
            )
            actual = laplacian(new)
            assert (
                tuple(
                    sum(c * x for c, x in zip(coeffs, (t, h, r, s)))
                    for coeffs in GRADIENTS
                )
                == actual[1:]
            )


def test_signed_far_support_and_joint_renewal_polygon(certificate):
    for row in certificate["rows"].values():
        g, local = row["gains2"]
        amplitude = 1 - g - local
        diameter, height = row["diameter"], row["height"]
        shift = certificate["law"]["factors"]["ZHIR_theta_shift_factor"] * PI / 4
        for h in (Q(0), diameter / 2, height):
            available = min(diameter - h, (amplitude * shift - (g + local) * h) / g)
            assert available >= 0
            assert available + h <= diameter
            assert h + g * available <= amplitude * (shift + h)
        for _, _, _, _, _, aa, ee, remainder in row["controls"]:
            # Piecewise-linear far support has its extrema at 0,1/2,1.
            observed = max(
                -aa * x + abs(ee) * min(x, 1 - x) for x in (Q(0), Q(1, 2), Q(1))
            )
            assert observed == remainder


def test_quadratic_maxima_have_independent_endpoint_or_vertex_certificates(certificate):
    for row in certificate["rows"].values():
        for poly, lower, upper, point, maximum, *_ in row["controls"]:

            def value(x):
                return sum(c * x**j for j, c in enumerate(poly))

            assert lower <= point <= upper
            assert maximum == value(point) >= max(value(lower), value(upper))
            if lower < point < upper:
                assert poly[2] < 0 and poly[1] + 2 * poly[2] * point == 0
                for boundary in (lower, upper):
                    assert (
                        maximum - value(boundary) == -poly[2] * (boundary - point) ** 2
                    )


def test_failed_cubic_envelope_is_exact_but_not_a_source(certificate):
    for row in certificate["rows"].values():
        assert len(row["bernstein"]) == 3
        for poly, left, right, coefficients in row["bernstein"]:
            # Four exact scalar identities determine a cubic, independently
            # of the power-to-Bernstein transformation used above.
            for t in (Q(0), Q(1, 3), Q(2, 3), Q(1)):
                x = left + (right - left) * t
                direct = sum(c * x**j for j, c in enumerate(poly))
                basis = sum(
                    coefficients[j] * comb(3, j) * t**j * (1 - t) ** (3 - j)
                    for j in range(4)
                )
                assert direct == basis
                assert min(coefficients) <= direct <= max(coefficients)
        assert row["unresolved"] > certificate["xi"]
