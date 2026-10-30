"""Finite exact controls for the new method; no field/root/transit evaluation."""
from fractions import Fraction as F
from pathlib import Path
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
checks = []


def check(name, value):
    if value is not True:
        raise AssertionError(name)
    checks.append(name)


eps, rho, eta = F(1, 10**32), F(1, 2048), F(1, 9000000)
lam, F0, sigma = F(1, 90), F(3, 5), F(1025)
edges = [(c + j, c + (j + 1) % 9) for c in (0, 9) for j in range(9)] + [(4, 13), (5, 14)]
degrees = tuple(sum(i in e for e in edges) for i in range(18))
check('all original edges and live mass', len(set(tuple(sorted(e)) for e in edges)) == 20 and sum(degrees) == 40 and {i for i, d in enumerate(degrees) if d == 3} == {4, 5, 13, 14})
incidence = [tuple(int(i == right) - int(i == left) for i in range(18)) for left, right in edges]
check('incidence and degree-weighted current conservation', all(sum(row) == 0 and sum(F(-b, d) * d for b, d in zip(row, degrees)) == 0 for row in incidence))
check('degree norm controls both source rows', 3 * (1 + 1) < 3**2)
alpha_coefficients = [2 * (j - 4) for j in range(9)] + [j - 4 for j in range(9)]
phi_coefficients = [0] * 9 + [1] * 9
check('nominal common mean coefficients', sum(d * a for d, a in zip(degrees, alpha_coefficients)) == 3 and sum(d * p for d, p in zip(degrees, phi_coefficients)) == 20)
for a, p in ((2, 0), (1, 1)):
    raw = [a * (j - 4) for j in range(9)]
    increments = [raw[j + 1] - raw[j] for j in range(8)] + [raw[0] - raw[8] + 9 * a]
    check('fixed cycle lift reconstruction', increments == [a] * 9)
check('distinct actual contact affine gaps', [(alpha_coefficients[r] - alpha_coefficients[l], phi_coefficients[r] - phi_coefficients[l]) for l, r in edges[-2:]] == [(0, 1), (-1, 1)])
check('positive contact work excludes old energy ceiling', F(1, 5) > F(1, 8))
check('source energy includes both internal residuals and contact', F(49, 200) + F(1, 500000) + 10 * eps**2 < F(1, 4))
check('port speed ceiling', F(2, 3) * (F(1, 500)**2 + F(7, 10)**2) < F(1, 3) < F0**2)
check('rational gamma envelope', 3069 < 1023 * F(25, 8) < 1023 * F(22, 7) < 3216)
check('eta ceiling', F(1, 1) / (1023 * F(25, 8))**2 < eta)
check('horizon square root ceiling', sigma / 4 < 17**2)
z0 = 3 * eps / 3069
v0 = 3 * eps + z0
v = v0 + 2 * eta * 17 / lam
z = (z0 + eta / lam * (F0 + 2 * v)) / (1 - 2 * eta / lam)
q = v + z
check('comparison denominator', 1 - 2 * eta / lam == 1 - F(1, 50000) > F(99, 100))
check('initial mixed-coordinate budgets', z0 < eps / 1000 and v0 < 4 * eps)
check('new mixed phase envelope', v < F(7, 20000))
check('new global scaled form envelope', z < F(1, 100000))
check('old stronger global bound does not follow from this envelope', z > F(1, 200000))
check('phase comparison and first-exit margin', q < F(9, 25000) < rho)
check('initial donor margin', F(25, 8) / 18 > F(1, 6))
check('analytic first quarter margin', F(1, 6) - F0 / 4 == F(1, 60) > rho)
check('tail decay exponent premise', lam / eta == 100000 > 512)
check('last-unit forcing bound', eta / lam * (2 * rho + 2 * q) < F(1, 51200000))
zend = F(1, 2**512 * 100000) + F(1, 51200000)
check('reconstructed endpoint form', 3216 * zend < F(1, 8192))
check('endpoint phase', q + rho < F(1, 1024))
r2 = F(1, 8192)**2 + F(1, 1024)**2
check('full-state endpoint storage and radius', r2 == F(65, 67108864) < F(1, 648000) < F(1, 12)**2)
check('basin boundary constant', F(1, 12)**2 / 4500 == F(1, 648000))
check('fixed reference schedule', 4096 * F(1, 4) == 1024 and 1024 + 1 == sigma)
check('static field and domain call bounds', 16 + 8 + 9 == 33 and 4096 * 33 == 135168 and 20 * 135168 == 2703360 and 4096 * 16 == 65536)
check('fixed artifact byte allocation', 256 * 1024**2 + 64 * 1024**2 == 320 * 1024**2)

receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
write_json_once(OUT / 'method-controls.json', {
    'passed': True, 'checks': checks,
    'source': receipt(Path(__file__)), 'protocol': receipt(OUT / 'controls-protocol.md'),
    'review': receipt(OUT / 'controls-review.md'),
    'actual_source_transits': 0, 'reference_steps': 0, 'root_solves': 0,
    'target_sign_assessments': 0, 'response_assessments': 0, 'historical_producer_replays': 0,
    'boundary': 'Exact algebra only: graph/source and newly derived comparison/tail design constants. No reference domain or endpoint has been numerically evaluated.'})
print({'passed': True, 'exact_controls': len(checks), 'reference_steps': 0})
