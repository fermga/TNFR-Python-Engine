"""Independent finite checks of the shared exact history-affine boundary."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tnfr.research import history_affine as h


def box(intervals):
    lo = [pair[0] for pair in intervals] + [0]
    hi = [pair[1] for pair in intervals] + [0]
    return [[0 if i == j else hi[i] - lo[j] for j in range(7)] for i in range(7)]


def raw_fixture():
    def interval(a, b):
        return box([(a, b)] + [(0, 0)] * 5)

    return dict(normalization=4, coefficient_vertices=[2, 7, 19],
                guards=[dict(guard_id=i, bounds=interval(a, b)) for i, (a, b) in enumerate(((-2, -1), (-2, 3), (-3, -2)))],
                root=dict(vertex=7, point=[3, 0, 0, 0, 0, 0], minimum='1'),
                drift_obligations=[dict(arc_index=9, source_vertex=7, target_vertex=19, guard_id=0, shift=[-1, 0, 0, 0, 0, 0]),
                                   dict(arc_index=3, source_vertex=7, target_vertex=2, guard_id=1, shift=[5, 0, 0, 0, 0, 0])],
                target_obligations=[dict(piece_index=8, vertex=19, guard_id=2)],
                provenance=dict(scope='algebraic fixture, no TNFR trajectory', finite_metadata=0.5))


def admit(raw=None):
    raw = raw_fixture() if raw is None else raw
    return h.admit_specification(raw, h.specification_digest(raw))


def candidate(spec):
    return dict(schema_version=1, specification_sha256=spec.digest,
                coefficients=[dict(vertex=2, gradient=[0] * 6, offset='3'),
                              dict(vertex=7, gradient=[4, 0, 0, 0, 0, 0], offset='0'),
                              dict(vertex=19, gradient=[4, 0, 0, 0, 0, 0], offset='1')],
                drift_proofs=[dict(arc_index=9, multipliers=[]), dict(arc_index=3, multipliers=[[0, 6, '1']])],
                target_proofs=[dict(piece_index=8, multipliers=[[0, 6, '1']])])


@pytest.mark.parametrize('value,expected', [
    (3, Fraction(3)), ('-2/3', Fraction(-2, 3)),
    (' 1.25e-2 ', Fraction(1, 80)), ('.5', Fraction(1, 2)), ('1.', Fraction(1)),
    (Fraction(5, 7), Fraction(5, 7)), ('-0', Fraction(0))])
def test_exact_rational_supported_bounded_values(value, expected):
    assert h.parse_rational(value, max_bits=16) == expected


@pytest.mark.parametrize('value', [
    True, False, 0.5, float('nan'), 'nan', 'inf', '1/0',
    '1/-2', '1_000', '1 + 2', '', '0x10', object(), '1e100000000000', 1 << 32, Fraction(1, 1 << 32)])
def test_exact_rational_rejects_ambiguous_or_over_budget(value):
    with pytest.raises(ValueError):
        h.parse_rational(value, max_bits=16)


@pytest.mark.parametrize('value', ['1e999999999999', '0e100000000', '1e-100000000', '9' * 10000, '1/' + '1' * 10000])
def test_text_budget_rejection_precedes_fraction_construction(monkeypatch, value):
    def forbidden(_value):
        raise AssertionError('Unbounded input reached Fraction')

    monkeypatch.setattr(h, 'Fraction', forbidden)
    with pytest.raises(ValueError, match='before parsing'):
        h.parse_rational(value, max_bits=16)


def test_admission_binds_metadata_and_freezes_independent_input():
    raw = raw_fixture()
    spec = admit(raw)
    raw['guards'][0]['bounds'][0][6] = 999
    raw['drift_obligations'][0]['shift'][0] = 999
    assert spec.guards[0][0][6] == -1 and spec.obligations[0].shift[0] == -1
    with pytest.raises(FrozenInstanceError):
        spec.normalization = 9
    with pytest.raises(TypeError):
        spec.guards[0][0][6] = 0
    with pytest.raises(TypeError):
        spec._columns[7] = 0
    with pytest.raises(ValueError, match='digest'):
        h.admit_specification(raw, spec.digest)
    forged = replace(spec, normalization=9)
    with pytest.raises(ValueError, match='admit_specification'):
        h.root_constraint(forged)


def test_pool_is_closed_once_and_not_in_each_candidate_check(monkeypatch):
    raw = raw_fixture()
    raw['guards'].append(dict(guard_id=3, bounds=deepcopy(raw['guards'][0]['bounds'])))
    calls = []
    original = h._closed

    def counted(bounds):
        calls.append(bounds)
        return original(bounds)

    monkeypatch.setattr(h, '_closed', counted)
    spec = admit(raw)
    assert len(calls) == 3 and spec.guards[0] is spec.guards[3]
    assert h.verify_candidate(spec, candidate(spec))['accepted']
    assert h.verify_candidate(spec, candidate(spec))['accepted']
    assert len(calls) == 3


@pytest.mark.parametrize('kind', [
    'bool_vertex', 'duplicate_vertex', 'bool_bound', 'unclosed_guard',
    'wrong_guard_id', 'duplicate_arc', 'outside_target', 'zero_scale', 'zero_root', 'bool_root_point',
    'missing_targets', 'unknown_guard', 'bad_shift'])
def test_specification_rejects_malformed_complete_premises(kind):
    raw = raw_fixture()
    if kind == 'bool_vertex':
        raw['coefficient_vertices'][0] = True
    elif kind == 'duplicate_vertex':
        raw['coefficient_vertices'][1] = 2
    elif kind == 'bool_bound':
        raw['guards'][0]['bounds'][0][0] = False
    elif kind == 'unclosed_guard':
        raw['guards'][0]['bounds'][1][2] = 1
    elif kind == 'wrong_guard_id':
        raw['guards'][0]['guard_id'] = 1
    elif kind == 'duplicate_arc':
        raw['drift_obligations'][1]['arc_index'] = 9
    elif kind == 'outside_target':
        raw['target_obligations'][0]['vertex'] = 999
    elif kind == 'zero_scale':
        raw['normalization'] = 0
    elif kind == 'zero_root':
        raw['root']['minimum'] = '0'
    elif kind == 'bool_root_point':
        raw['root']['point'][0] = True
    elif kind == 'missing_targets':
        raw['target_obligations'] = []
    elif kind == 'unknown_guard':
        raw['drift_obligations'][0]['guard_id'] = 99
    else:
        raw['drift_obligations'][0]['shift'][0] = 0.5
    with pytest.raises(ValueError):
        admit(raw)


def test_resource_limits_are_explicit_and_do_not_accept_a_prefix():
    raw = raw_fixture()
    for kwargs in (dict(max_vertices=2), dict(max_guards=2), dict(max_obligations=2), dict(max_integer_bits=2)):
        with pytest.raises(ValueError):
            h.admit_specification(raw, h.specification_digest(raw), **kwargs)
    with pytest.raises(ValueError):
        h.specification_digest(dict(value=10 ** 6000))
    with pytest.raises(ValueError):
        h.specification_digest(dict(value=float('inf')))


def test_shared_rows_and_forms_match_direct_signed_shifted_potentials():
    raw = raw_fixture()
    raw['drift_obligations'].append(dict(arc_index=17, source_vertex=7, target_vertex=7, guard_id=1, shift=[-1, 2, 0, 0, 0, 0]))
    spec = admit(raw)
    records = [dict(vertex=v, gradient=[str(Fraction((v + 3) * (i + 1) % 7 - 3, i + 1)) for i in range(6)], offset=str(Fraction(v, 11))) for v in spec.vertices]
    blocks = h.parse_coefficients(spec, records)
    by_vertex = {v: (a, b) for v, a, b in blocks.records}

    def potential(v, point):
        gradient, offset = by_vertex[v]
        return offset + sum(a * k / spec.normalization for a, k in zip(gradient, point))

    for obligation in spec.obligations:
        bounds = spec.guards[obligation.guard_id]
        for x in range(-bounds[6][0], bounds[0][6] + 1):
            point = (x, 0, 0, 0, 0, 0)
            direct = -potential(obligation.source_vertex, point)
            if obligation.kind == 'drift':
                direct += potential(obligation.target_vertex, tuple(k + s for k, s in zip(point, obligation.shift)))
            linear, constant = h.affine_form(spec, blocks, obligation)
            row = h.point_constraint(spec, obligation, point)
            assert constant + sum(c * k for c, k in zip(linear, point)) == direct
            assert sum(value * blocks.flat[column] for column, value in row) == direct
    row, minimum = h.root_constraint(spec)
    assert minimum == 1 and sum(value * blocks.flat[column] for column, value in row) == potential(7, spec.root_point)
    assert h.affine_form(spec, blocks, spec.obligations[2])[0] == (0,) * 6  # Self-loop gradients cancel.
    with pytest.raises(ValueError, match='outside'):
        h.point_constraint(spec, spec.obligations[0], (999, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match='belong'):
        h.affine_form(spec, blocks, replace(spec.obligations[0], shift=(0,) * 6))
    with pytest.raises(ValueError, match='parsed'):
        h.affine_form(spec, replace(blocks, flat=(Fraction(0),) * 21), spec.obligations[0])
    with pytest.raises(ValueError, match='parsed'):
        h.affine_form(admit(raw), blocks, spec.obligations[0])


def test_complete_candidate_keeps_nonzero_root_shift_and_target_scope():
    spec = admit()
    result = h.verify_candidate(spec, candidate(spec))
    assert result['accepted'] and result['root_value'] == '3'
    assert (result['checked_drift_obligations'], result['checked_target_obligations']) == (2, 1)
    assert not result['whole_C6_first_exit_label_excluded'] and not result['original_target_coverage_verified']
    blocks = h.parse_coefficients(spec, candidate(spec)['coefficients'])
    assert [h.exact_minimum(spec, item, blocks)['minimum'] for item in spec.obligations] == ['0', '0', '1']


@pytest.mark.parametrize('kind', [
    'root_minimum', 'root_fraction', 'normalization', 'guard_pool',
    'root_point', 'obligation_kind', 'obligation_shift', 'obligation_guard'])
def test_reflection_mutation_cannot_keep_old_specification_admission(kind):
    spec = admit()
    payload = candidate(spec)
    if kind == 'root_minimum':
        object.__setattr__(spec, 'root_minimum', Fraction(0))
    elif kind == 'root_fraction':
        object.__setattr__(spec.root_minimum, '_numerator', 0)
    elif kind == 'normalization':
        object.__setattr__(spec, 'normalization', 1)
    elif kind == 'guard_pool':
        object.__setattr__(spec, 'guards', tuple(reversed(spec.guards)))
    elif kind == 'root_point':
        object.__setattr__(spec, 'root_point', (0,) * 6)
    elif kind == 'obligation_kind':
        object.__setattr__(spec.obligations[0], 'kind', 'target')
    elif kind == 'obligation_shift':
        object.__setattr__(spec.obligations[0], 'shift', (0,) * 6)
    else:
        object.__setattr__(spec.obligations[0], 'guard_id', 1)
    with pytest.raises(ValueError, match='mutated'):
        h.verify_candidate(spec, payload, collect_failures=True)


@pytest.mark.parametrize('kind', ['flat', 'records', 'flat_fraction', 'record_fraction'])
def test_coefficient_owner_snapshot_rejects_nested_reflection_mutation(kind):
    spec = admit()
    blocks = h.parse_coefficients(spec, candidate(spec)['coefficients'])
    if kind == 'flat':
        object.__setattr__(blocks, 'flat', (Fraction(0),) * 21)
    elif kind == 'records':
        object.__setattr__(blocks, 'records', ())
    elif kind == 'flat_fraction':
        object.__setattr__(blocks.flat[7], '_numerator', 0)
    else:
        object.__setattr__(blocks.records[1][1][0], '_numerator', 0)
    with pytest.raises(ValueError, match='mutated'):
        h.affine_form(spec, blocks, spec.obligations[0])


def test_fraction_input_is_not_aliased_into_admitted_coefficients():
    spec = admit()
    records = candidate(spec)['coefficients']
    supplied = Fraction(4)
    records[1]['gradient'][0] = supplied
    blocks = h.parse_coefficients(spec, records)
    object.__setattr__(supplied, '_numerator', 99)
    assert blocks.flat[7] == 4
    assert h.affine_form(spec, blocks, spec.obligations[0]) == ((0,) * 6, 0)


def test_specification_admission_uses_the_same_private_bytes_as_digest(monkeypatch):
    raw = raw_fixture()
    binding = h.specification_digest(raw)
    original = h._canonical_bytes

    def race_after_canonicalization(value):
        encoded = original(value)
        value['root']['minimum'] = '0'
        value['drift_obligations'][0]['shift'][0] = 900
        return encoded

    monkeypatch.setattr(h, '_canonical_bytes', race_after_canonicalization)
    spec = h.admit_specification(raw, binding)
    assert spec.root_minimum == 1 and spec.obligations[0].shift[0] == -1
    assert h.verify_candidate(spec, candidate(spec))['accepted']


def test_candidate_digest_and_schema_keys_never_dispatch_foreign_equality():
    class ForeignDigest:
        def __eq__(self, other):
            raise AssertionError('Foreign equality was dispatched')

    class ForeignKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            raise AssertionError('Foreign key equality was dispatched')

    spec = admit()
    payload = candidate(spec)
    payload['specification_sha256'] = ForeignDigest()
    with pytest.raises(ValueError):
        h.verify_candidate(spec, payload)
    payload = candidate(spec)
    version = payload.pop('schema_version')
    payload[ForeignKey('schema_version')] = version
    with pytest.raises(ValueError):
        h.verify_candidate(spec, payload)


@pytest.mark.parametrize('kind', [
    'missing_block', 'reordered_blocks', 'foreign_digest', 'missing_proof',
    'reordered_proofs', 'duplicate_dual', 'negative_dual', 'bool_index', 'float_coefficient', 'overlong_coefficient'])
def test_malformed_candidates_never_become_collected_mathematical_failures(kind):
    spec = admit()
    payload = candidate(spec)
    if kind == 'missing_block':
        payload['coefficients'].pop()
    elif kind == 'reordered_blocks':
        payload['coefficients'].reverse()
    elif kind == 'foreign_digest':
        payload['specification_sha256'] = '0' * 64
    elif kind == 'missing_proof':
        payload['target_proofs'] = []
    elif kind == 'reordered_proofs':
        payload['drift_proofs'].reverse()
    elif kind == 'duplicate_dual':
        payload['target_proofs'][0]['multipliers'] *= 2
    elif kind == 'negative_dual':
        payload['target_proofs'][0]['multipliers'][0][2] = '-1'
    elif kind == 'bool_index':
        payload['target_proofs'][0]['multipliers'][0][0] = False
    elif kind == 'float_coefficient':
        payload['coefficients'][0]['offset'] = 3.0
    else:
        payload['coefficients'][0]['offset'] = '0e999999999999999999'
    with pytest.raises(ValueError):
        h.verify_candidate(spec, payload, collect_failures=True)


def test_complete_failure_collection_does_not_skip_later_obligations():
    spec = admit()
    payload = candidate(spec)
    for record in payload['coefficients']:
        record['gradient'], record['offset'] = [0] * 6, '0'
    for proof in payload['drift_proofs'] + payload['target_proofs']:
        proof['multipliers'] = []
    zero = h.verify_candidate(spec, payload, collect_failures=True)
    assert not zero['accepted'] and [f['kind'] for f in zero['failures']] == ['root']
    for record in payload['coefficients']:
        record['offset'] = '1'
    one = h.verify_candidate(spec, payload, collect_failures=True)
    assert not one['accepted'] and [f['kind'] for f in one['failures']] == ['target']
    assert one['checked_drift_obligations'] == 2 and one['checked_target_obligations'] == 1


def test_exact_extrema_equal_independent_finite_integer_oracle():
    intervals = [(-2, 2), (-1, 1)] + [(0, 0)] * 4
    bounds = box(intervals)
    points = list(product(*(range(a, b + 1) for a, b in intervals)))
    for first, second in product((-2, 0, 3), (-3, 0, 2)):
        coefficients = (Fraction(first, 3), Fraction(second, 5), 0, 0, 0, 0)
        constant = Fraction(2, 7)
        result = h.exact_affine_minimum(bounds, coefficients, constant)
        expected = min(constant + sum(c * k for c, k in zip(coefficients, point)) for point in points)
        assert Fraction(result['minimum']) == expected
        assert tuple(result['attaining_point']) in points
        assert result['nonnegative_certified'] == (expected >= 0)
        assert result['exact_primal_dual_equality_checked']


@pytest.mark.parametrize('kind', ['primal', 'objective', 'dual', 'boolean_flow'])
def test_owner_receipt_is_independently_checked(monkeypatch, kind):
    import tnfr.physics.c6_carried_excursion as owner

    original = owner._linear_extremum

    def corrupt(*args):
        receipt = original(*args)
        fields = dict(value=receipt.value, point=receipt.point, dual_flow=receipt.dual_flow)
        if kind == 'primal':
            fields['point'] = (99, 0, 0, 0, 0, 0)
        elif kind == 'objective':
            fields['value'] += 1
        elif kind == 'dual':
            fields['dual_flow'] = ()
        else:
            fields['dual_flow'] = tuple((i, j, True) for i, j, _a in fields['dual_flow'])
        return SimpleNamespace(**fields)

    monkeypatch.setattr(owner, '_linear_extremum', corrupt)
    with pytest.raises((ValueError, RuntimeError)):
        h.exact_affine_minimum(box([(-1, 1)] + [(0, 0)] * 5), [1, 0, 0, 0, 0, 0], 0)


@pytest.mark.parametrize('flag', ['-O', '-OO'])
def test_optimized_python_preserves_explicit_admission_and_refuses_exact_owner(flag):
    raw = raw_fixture()
    proposal = candidate(admit(raw))
    code = f'''
from fractions import Fraction
from tnfr.research.history_affine import (
    parse_rational, exact_affine_minimum, admit_specification,
    specification_digest, verify_candidate,
)
raw = {raw!r}
proposal = {proposal!r}
spec = admit_specification(raw, specification_digest(raw))
if not verify_candidate(spec, proposal)['accepted']:
    raise SystemExit('optimized full verification rejected valid fixture')
object.__setattr__(spec, 'root_minimum', Fraction(0))
try:
    verify_candidate(spec, proposal)
except ValueError:
    pass
else:
    raise SystemExit('optimized verification accepted mutated root')
try:
    parse_rational(True)
except ValueError:
    pass
else:
    raise SystemExit('optimized admission accepted bool')
try:
    exact_affine_minimum([[0] * 7 for _ in range(7)], [0] * 6, 0)
except RuntimeError as error:
    if 'optimized Python' not in str(error):
        raise
else:
    raise SystemExit('optimized extrema were not refused')
print('explicit admission active; exact owner refused')
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
    result = subprocess.run([sys.executable, flag, '-c', code], env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'explicit admission active; exact owner refused' in result.stdout
