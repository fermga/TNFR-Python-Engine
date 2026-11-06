"""Bounded documentary/arithmetic audit of the new arrival attempt.

No field, trigonometric bound, Taylor jet, root or trajectory is re-evaluated.
Stored primitive enclosure claims retain the reviewed producer's mathematical
premises. Receipts associate bytes; this is not adversarial authentication.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from fractions import Fraction as Q
import io
import math
from pathlib import Path
import sys
import zipfile

from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._rational_interval import I
from ..mathematics._validated_metric import (
    METRIC_TAYLOR_METHOD, ValidatedMetricTaylorStep, _AUTONOMOUS_PREMISES,
    _growth_upper, _metric_coordinate_projection_factors, _metric_error_upper,
)
from ..mathematics._validated_metric_inclusion import (
    METRIC_INCLUSION_METHOD, ValidatedMetricInclusionStep, _PREMISES,
    metric_forcing_gain_upper,
)
from ..utils.io import json_loads
from .artifact_io import (decode_exact_tree, exact_record, file_receipt,
                          read_bytes_bounded, sha256_bytes, validate_artifact_path,
                          verify_archive_members)
from . import formed_two_port_arrival as model_owner
from . import formed_two_port_arrival_run as run_owner
from .formed_two_port_run import _contained, _json_bytes


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _shape(value, expected, name):
    _require(type(value) is dict and set(value) == set(expected), f"invalid {name} schema")


def _dataclass_shape(value, cls, name):
    _shape(value, (x.name for x in fields(cls)), name)


def _math_tree(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            _math_tree(item)
    elif isinstance(value, dict):
        _shape(value, ('lo', 'hi'), 'interval')
        _interval(value)
    else:
        exact_record(value)


def _interval(value):
    _shape(value, ('lo', 'hi'), 'interval')
    lo, hi = exact_record(value['lo']), exact_record(value['hi'])
    _require(lo <= hi, 'reversed interval')
    result = I(lo, hi)
    _require(result.lo == lo and result.hi == hi, 'interval endpoints not retained exactly')
    return result


def _intervals(values, size):
    _require(isinstance(values, (tuple, list)) and len(values) == size, 'interval dimension')
    return tuple(_interval(x) for x in values)


def _rational_row(values, size):
    _require(isinstance(values, (tuple, list)) and len(values) == size, 'rational dimension')
    return tuple(exact_record(x) for x in values)


def _receipt_bytes(root, receipt, *, limit):
    _shape(receipt, ('path', 'bytes', 'sha256'), 'receipt')
    _require(type(receipt['bytes']) is int and 0 <= receipt['bytes'] <= limit,
             'receipt byte budget')
    name = validate_artifact_path(receipt['path'])
    path = _contained(root, name)
    data = read_bytes_bounded(path, max_bytes=receipt['bytes'])
    _require(len(data) == receipt['bytes'] and sha256_bytes(data) == receipt['sha256'],
             'receipt bytes differ')
    return data


def _decode(data):
    return decode_exact_tree(json_loads(data))


def audit_inclusion_certificate(record, model, center, radius, time):
    """Rebuild stored algebra, conditional on the producer's primitive bounds.

    ``model`` supplies an exact chart and graph; this helper evaluates no drift.
    The outer attempt auditor binds it to the actual fixed model.
    """
    _dataclass_shape(record, ValidatedMetricInclusionStep, 'inclusion certificate')
    for key in ('forcing_bound', 'effective_forcing_bound', 'forcing_gain_upper_bound',
                'forcing_radius_upper_bound', 'endpoint_center', 'endpoint_radius', 'endpoint'):
        _math_tree(record[key])
    _require(record['method'] == METRIC_INCLUSION_METHOD
             and record['conditional_premises'] == _PREMISES
             and record['forcing_refined'] is True, 'inclusion method/premises differ')
    nominal = record['nominal_step']
    _dataclass_shape(nominal, ValidatedMetricTaylorStep, 'nominal certificate')
    for key in set(nominal)-{'growth_certificate', 'method', 'conditional_premises', 'prescribed_time_interval'}:
        _math_tree(nominal[key])
    n, metric = model.chart.dimension, model.metric
    projections = _metric_coordinate_projection_factors(metric)
    _require(nominal['time'] == time and nominal['duration'] == run_owner.STEP_SIZE,
             'step clock differs')
    _require(type(nominal['order']) is int and nominal['order'] == run_owner.ORDER,
             'step order differs')
    _require(nominal['metric'] == metric and nominal['initial_center'] == center
             and nominal['initial_radius'] == radius, 'incoming metric ball differs')
    _require(nominal['coordinate_projection_factors'] == projections,
             'metric coordinate projection differs')
    _require(nominal['method'] == METRIC_TAYLOR_METHOD
             and nominal['conditional_premises'] == _AUTONOMOUS_PREMISES
             and nominal['growth_certificate'] is None
             and nominal['prescribed_time_interval'] is None, 'nominal method differs')
    initial = tuple(I(x-p*radius, x+p*radius) for x, p in zip(center, projections))
    _require(_intervals(nominal['initial_box'], n) == initial, 'initial projection differs')
    tube = _intervals(nominal['tube'], n)
    _require(all(t.lo <= b.lo <= b.hi <= t.hi for t, b in zip(tube, initial)),
             'initial box not contained in admitted tube')
    _require(nominal['picard_interior_margin'] > 0
             and len(nominal['domain_lower_bounds']) > 0
             and min(nominal['domain_lower_bounds']) > 0, 'strict tube premise missing')
    proof_tube = _intervals(record['growth_proof_tube'], n)
    _require(all(p.lo <= t.lo <= t.hi <= p.hi for t, p in zip(tube, proof_tube)),
             'tube exceeds growth proof')
    proof = record['external_growth_certificate']
    _dataclass_shape(proof, model_owner.SignedGraphGrowth, 'signed graph proof')
    for key in set(proof)-{'certified', 'reason', 'prescribed_time_interval'}:
        _math_tree(proof[key])
    _require(proof['certified'] is True and proof['reason'] is None
             and proof['prescribed_time_interval'] is None, 'growth proof unavailable')
    _require(_intervals(proof['tube'], n) == proof_tube and proof['metric'] == metric,
             'growth proof associations differ')
    _require(proof['rate_bounds'] == run_owner.RATE_BOUNDS
             and type(proof['bisections_requested']) is int
             and proof['bisections_requested'] == run_owner.RATE_BISECTIONS
             and type(proof['bisections_performed']) is int
             and 0 <= proof['bisections_performed'] <= run_owner.RATE_BISECTIONS,
             'growth search policy differs')
    weights = _rational_row(proof['edge_cosine_lower_bounds'], len(model.reference.oriented_edges))
    _require(all(-1 <= x <= 1 for x in weights), 'cosine lower bounds outside range')
    basis = model.chart.basis
    vectors = tuple(tuple(basis[j][k]-basis[i][k] for k in range(n))
                    for i, j in model.reference.oriented_edges)
    stiffness = tuple(tuple(sum(w*v[i]*v[j] for w, v in zip(weights, vectors))
                            for j in range(n)) for i in range(n))
    rate = exact_record(proof['growth_rate_upper_bound'])
    _require(run_owner.RATE_BOUNDS[0] <= rate <= run_owner.RATE_BOUNDS[1],
             'growth rate outside fixed bracket')
    low, high = run_owner.RATE_BOUNDS
    if proof['bisections_performed'] == 0:
        _require(rate == low, 'zero-bisection rate differs from certified lower endpoint')
    else:
        grid = (rate-low)*2**run_owner.RATE_BISECTIONS/(high-low)
        _require(proof['bisections_performed'] == run_owner.RATE_BISECTIONS
                 and rate > low and grid.denominator == 1,
                 'growth rate/count differs from fixed bisection grid')
    slack = tuple(tuple(rate*metric[i][j]+stiffness[i][j] for j in range(n))
                  for i in range(n))
    _require(proof['stiffness_lower_bound'] == stiffness and proof['slack_matrix'] == slack
             and exact_symmetric_semidefinite(slack), 'growth PSD slack differs or fails')
    _require(nominal['growth_rate_upper_bound'] == rate, 'nominal growth rate differs')
    growth = _growth_upper(rate*run_owner.STEP_SIZE)
    _require(nominal['growth_factor_upper_bound'] == growth, 'growth factor differs')
    polynomial = _intervals(nominal['center_polynomial_bounds'], n)
    remainder = _intervals(nominal['local_remainder_bounds'], n)
    solution = tuple(a+b for a, b in zip(polynomial, remainder))
    _require(_intervals(nominal['center_solution_endpoint'], n) == solution,
             'center endpoint sum differs')
    next_center = tuple(x.midpoint for x in solution)
    errors = tuple(x.radius for x in solution)
    local = _metric_error_upper(metric, errors)
    propagated = I(growth*radius).hi
    nominal_radius = I(propagated+local).hi
    _require(nominal['endpoint_center'] == next_center
             and nominal['local_coordinate_error_bounds'] == errors
             and nominal['local_metric_error_upper_bound'] == local
             and nominal['propagated_initial_radius'] == propagated
             and nominal['endpoint_radius'] == nominal_radius, 'nominal radius arithmetic differs')
    nominal_endpoint = tuple(I(x-p*nominal_radius, x+p*nominal_radius)
                             for x, p in zip(next_center, projections))
    _require(_intervals(nominal['endpoint'], n) == nominal_endpoint,
             'nominal endpoint projection differs')
    effective = exact_record(record['effective_forcing_bound'])
    _require(record['forcing_bound'] == Q(1, 7000)
             and 0 <= effective <= Q(1, 7000), 'forcing bounds differ')
    gain = metric_forcing_gain_upper(rate, run_owner.STEP_SIZE)
    forcing_radius = I(effective*gain).hi
    next_radius = I(nominal_radius+forcing_radius).hi
    endpoint = tuple(I(x-p*next_radius, x+p*next_radius)
                     for x, p in zip(next_center, projections))
    _require(record['forcing_gain_upper_bound'] == gain
             and record['forcing_radius_upper_bound'] == forcing_radius
             and record['endpoint_center'] == next_center
             and record['endpoint_radius'] == next_radius
             and _intervals(record['endpoint'], n) == endpoint,
             'actual forcing/radius arithmetic differs')
    return next_center, next_radius


def audit_terminal_record(record, radius, time, target_lower):
    _dataclass_shape(record, model_owner.ArrivalTerminal, 'terminal decision')
    for key in set(record)-{'status', 'reasons', 'scope'}:
        exact_record(record[key])
    _require(record['time'] == time and record['target_storage_lower_bound'] == target_lower,
             'terminal time/target association differs')
    rebuilt = model_owner._terminal_arithmetic(
        time, record['phase_distance_upper_bound'], radius,
        record['center_field_norm_upper_bound'], record['phase_potential_upper_bound'], target_lower)
    _require(asdict(rebuilt) == record, 'terminal arithmetic differs')
    return rebuilt.status


def _closing_success(snapshot, returned):
    _require(type(returned) is dict and set(returned) == set(snapshot), 'closing report schema differs')
    mutable = {'record_stage', 'supervisor_elapsed_seconds',
               'terminal_write_deadline_verified', 'total_budget_exhausted'}
    _require(all(returned[k] == snapshot[k] for k in set(snapshot)-mutable),
             'closing report differs from supervisor snapshot')
    _require(snapshot['record_stage'] == 'pre_write_snapshot_not_final_attempt_verdict'
             and returned['record_stage'] == 'post_write_result_required_for_attempt_admission',
             'post-write supervisor result required')
    for key in ('passed', 'job_assigned_before_resume', 'resumed',
                'worker_stopped_observed', 'cleanup_complete', 'terminal_write_deadline_verified'):
        _require(returned.get(key) is True, f'closing {key} not verified')
    for key in ('watchdog_fired', 'total_budget_exhausted'):
        _require(returned.get(key) is False, f'closing {key} invalidates attempt')
    _require(returned.get('status') == 'success' and type(returned.get('exit_code')) is int
             and returned['exit_code'] == 0 and returned.get('watchdog_error') is None,
             'supervisor did not finish successfully')
    elapsed = returned['supervisor_elapsed_seconds']
    prior = snapshot['supervisor_elapsed_seconds']
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed)
             and type(prior) in (int, float) and math.isfinite(prior)
             and 0 <= prior <= elapsed < 3600
             and returned['wall_seconds_limit'] == 3600
             and returned['process_memory_limit_bytes'] == 1024**3,
             'closing resource declaration differs')


def _current_audit_runtime():
    """Snapshot the logical environment and actual CPython launch image.

    The reviewed Windows supervisor launches ``_base_executable`` directly,
    retaining the virtual environment through its checked bootstrap. Its OS
    image therefore need not equal the logical ``sys.executable`` redirector.
    """
    return {
        'python_executable': str(Path(sys.executable).resolve()),
        'python_prefix': str(Path(sys.prefix).resolve()),
        'python_version': sys.version,
        'package_root': str(Path(__file__).resolve().parents[1]),
        'base_executable': str(Path(getattr(sys, '_base_executable', sys.executable)).resolve()),
    }


def _admit_runtime_association(frozen, attempt, image):
    """Bind a retained launch to this same declared auditor environment.

    The archived reviewed supervisor/worker bootstrap supplies the historical
    launch premise; current paths are an environment association, not a claim
    that interpreter binaries are cryptographically authenticated or unchanged.
    A relocated/different environment is refused, not inferred from filenames.
    """
    current = _current_audit_runtime()
    _shape(current, (*frozen, 'base_executable'), 'current audit runtime')
    _require(all(type(x) is str and x for x in current.values()),
             'invalid current audit runtime')
    _require(current['python_version'] == frozen['python_version']
             and all(Path(current[key]).resolve() == Path(frozen[key]).resolve()
                     for key in ('python_executable', 'python_prefix', 'package_root')),
             'auditor environment differs from frozen runtime')
    _require(Path(attempt['interpreter']).resolve() == Path(frozen['python_executable']).resolve()
             and Path(attempt['prefix']).resolve() == Path(frozen['python_prefix']).resolve(),
             'attempt logical environment differs from frozen runtime')
    _require(type(image) is str and image
             and Path(image).resolve() == Path(current['base_executable']).resolve(),
             'supervisor image differs from admitted base interpreter')
    return current


def audit_formed_two_port_arrival_attempt(directory, *, post_write_supervisor_result):
    """Audit a successful closed attempt; unavailable work is never nonarrival.

    The caller must preserve the actual dictionary returned after the supervisor
    wrote execution.json. Its pre-write snapshot alone cannot authorize success.
    Mathematical primitive evaluations are explicitly not re-executed here.
    Run this audit in the same declared interpreter, prefix, version and package
    root; its base executable associates the reviewed direct-image launcher.
    """
    root = Path(directory).resolve(strict=True)
    def read(name, limit=run_owner.AUX_BYTES):
        return _decode(read_bytes_bounded(_contained(root, name), max_bytes=limit))
    snapshot = read('execution.json', 512*1024)
    _closing_success(snapshot, post_write_supervisor_result)
    attempt = _decode(_receipt_bytes(root, post_write_supervisor_result['attempt'], limit=512*1024))
    _receipt_bytes(root, post_write_supervisor_result['log'], limit=1024**2)
    admission = read('admission.json')
    _shape(admission, ('source', 'model', 'actual_source', 'target', 'initial_decision',
                       'full_law_and_original_source_preserved'), 'admission')
    _require(admission['full_law_and_original_source_preserved'] is True, 'full law premise missing')
    association = admission['source']
    _shape(association, ('protocol', 'archive', 'head', 'worker_entry',
                        'target_evidence', 'prefix_evidence'), 'source association')
    protocol = _decode(_receipt_bytes(root, association['protocol'], limit=1024**2))
    _shape(protocol, ('schema', 'policy', 'head', 'runtime', 'sources', 'archive',
                      'reviews', 'controls', 'worker_entry'), 'protocol')
    _require(protocol['schema'] == run_owner.SCHEMA
             and _json_bytes(protocol['policy']) == _json_bytes(run_owner.declared_policy())
             and protocol['head'] == association['head']
             and protocol['worker_entry'] == association['worker_entry'], 'frozen policy differs')
    _shape(protocol['runtime'], ('python_executable', 'python_prefix', 'python_version', 'package_root'),
           'frozen runtime')
    _require(all(type(x) is str and x for x in protocol['runtime'].values()), 'invalid runtime metadata')
    _shape(attempt, ('script', 'arguments', 'cwd', 'interpreter', 'prefix', 'wall_seconds',
                    'closing_reserve_seconds', 'worker_seconds_budget', 'process_memory_bytes',
                    'memory_metric', 'log_limit_bytes', 'one_worker'), 'supervisor attempt')
    cwd = Path(attempt['cwd']).resolve()
    arguments = attempt['arguments']
    _require(isinstance(arguments, tuple) and len(arguments) == 6
             and all(type(x) is str for x in arguments)
             and arguments[::2] == ('--root', '--protocol', '--output')
             and Path(arguments[1]).resolve() == cwd and Path(arguments[5]).resolve() == root
             and Path(arguments[3]).resolve().is_relative_to(cwd), 'attempt arguments differ')
    _require(Path(attempt['script']).resolve() == (cwd/protocol['worker_entry']['path']).resolve()
             and (cwd/'src/tnfr').resolve() == Path(protocol['runtime']['package_root']).resolve(),
             'attempt execution does not match frozen runtime/entry')
    audit_runtime = _admit_runtime_association(
        protocol['runtime'], attempt, post_write_supervisor_result['image'])
    _require(type(attempt['wall_seconds']) is int and attempt['wall_seconds'] == 3600
             and type(attempt['process_memory_bytes']) is int and attempt['process_memory_bytes'] == 1024**3
             and type(attempt['log_limit_bytes']) is int and attempt['log_limit_bytes'] == 1024**2
             and attempt['one_worker'] is True
             and attempt['closing_reserve_seconds'] == 10
             and attempt['worker_seconds_budget'] == 3590, 'attempt resource policy differs')
    start = read('worker-start.json', 1024**2)
    _shape(start, ('schema', 'job', 'policy'), 'worker start')
    _require(start['schema'] == run_owner.SCHEMA
             and _json_bytes(start['policy']) == _json_bytes(run_owner.declared_policy()),
             'worker start policy differs')
    archive = _receipt_bytes(root, association['archive'], limit=run_owner.AUX_BYTES)
    _require(all(association['archive'][key] == protocol['archive'][key]
                 for key in ('bytes', 'sha256')), 'copied archive differs from frozen receipt')
    rows = protocol['sources']
    _require(isinstance(rows, tuple) and len(rows) <= 10000, 'invalid source inventory')
    names = [x['path'] for x in rows]
    _require(len(names) == len(set(names)), 'duplicate source inventory')
    source_rows = {x['path']: x for x in rows}
    _require(source_rows.get(protocol['worker_entry']['path']) == protocol['worker_entry'],
             'worker entry absent from frozen inventory')
    _require((run_owner._PREFIX_FILES | run_owner._TARGET_FILES | run_owner._OWNERS) <= set(names),
             'missing frozen source evidence')
    verify_archive_members(io.BytesIO(archive), {x['path']: x['sha256'] for x in rows},
                           max_bytes=run_owner.AUX_BYTES)
    reviews = protocol['reviews']
    _require(isinstance(reviews, tuple) and len(reviews) >= 2
             and len({x['path'] for x in reviews}) == len(reviews), 'independent review associations missing')
    for index, receipt in enumerate(reviews):
        _receipt_bytes(root, {**receipt, 'path': f'review-{index:02d}.md'}, limit=run_owner.AUX_BYTES)
    controls = _decode(_receipt_bytes(root, {**protocol['controls'], 'path': 'controls.json'},
                                     limit=run_owner.AUX_BYTES))
    _require(controls.get('passed') is True and type(controls.get('reserved_actual_steps')) is int
             and controls['reserved_actual_steps'] == 0, 'source-independent controls missing')
    _require({x['path']: x for x in association['prefix_evidence']}
             == {x: source_rows[x] for x in run_owner._PREFIX_FILES}, 'prefix association differs')
    _require({x['path']: x for x in association['target_evidence']}
             == {x: source_rows[x] for x in run_owner._TARGET_FILES}, 'target association differs')
    model = model_owner.build_formed_two_port_arrival_model()  # algebra/constants only
    _require(_json_bytes(asdict(model)) == _json_bytes(admission['model']), 'fixed model differs')
    source = admission['actual_source']
    _dataclass_shape(source, model_owner.ArrivalSource, 'actual source')
    for key in set(source)-{'evidence_receipts', 'scope'}:
        _math_tree(source[key])
    with zipfile.ZipFile(io.BytesIO(archive)) as saved:
        old = _decode(saved.read(f'{run_owner._PREFIX}/attempt/steps/0003.json'))['certificate']
    old_center = _rational_row(old['endpoint_center'], 18)
    old_radius = exact_record(old['endpoint_radius'])
    center = model.chart.project(old_center)
    radius = old_radius+Q(1, 45000)
    _require(0 <= old_radius < Q(1, 2**26) and source['time'] == 1
             and source['center'] == center and source['radius'] == radius
             and source['prefix_reference_radius'] == old_radius
             and source['complete_source_bridge_radius'] == Q(1, 45000), 'initial source bridge differs')
    if 'reference_center_weighted_mean' in source:
        _require(source['reference_center_weighted_mean']
                 == sum(d*x for d, x in zip(model.chart.degrees, old_center))/40
                 and source['prefix_radius_premise_margin'] == Q(1, 2**26)-old_radius,
                 'source projection premise differs')
    expected_prefix = {f"{run_owner._PREFIX}/{x['path']}": {**x, 'path': f"{run_owner._PREFIX}/{x['path']}"}
                       for x in source['evidence_receipts']}
    _require(expected_prefix == {x['path']: x for x in association['prefix_evidence']},
             'consumed prefix receipts differ')
    target = admission['target']
    _dataclass_shape(target, model_owner.ArrivalTarget, 'target')
    _dataclass_shape(target['retained_target'], model_owner.reference_owner.FormedTwoPortTarget,
                     'retained target')
    for key in ('target_phases', 'target_deviations', 'target_turn_affine_coefficients',
                'acute_margin_lower_bound'):
        _math_tree(target['retained_target'][key])
    _intervals(target['retained_target']['target_phases'], 18)
    _intervals(target['retained_target']['target_deviations'], 18)
    coefficients = target['retained_target']['target_turn_affine_coefficients']
    _require(isinstance(coefficients, tuple) and len(coefficients) == 18,
             'target affine coefficient dimension')
    for row in coefficients:
        _rational_row(row, 3)
    _require(target['retained_target']['acute_margin_lower_bound'] > 0, 'target acute margin missing')
    _intervals(target['relative_deviations'], 18)
    _require({x['path']: x for x in target['evidence_receipts']}
             == {x['path']: x for x in association['target_evidence']}
             and target['retained_target']['evidence_receipts'] == target['evidence_receipts'],
             'consumed target receipts differ')
    target_lower = _interval(target['energy_interval']).lo
    time = run_owner.START_TIME
    status = audit_terminal_record(admission['initial_decision'], radius, time, target_lower)
    last_decision = admission['initial_decision']
    terminal = read('worker-result.json', 1024**2)
    expected_terminal = ('status', 'completed_steps', 'endpoint_time', 'source', 'last_complete_step',
                         'decision', 'step_evidence_reserved_bytes',
                         'conditional_on_successful_post_write_supervisor_return', 'boundary')
    _shape(terminal, expected_terminal, 'worker terminal')
    count = terminal['completed_steps']
    _require(type(count) is int and 0 <= count <= run_owner.STEP_COUNT, 'completed step count')
    expected_names = {f'{i:04d}.json' for i in range(count)}
    step_dir = _contained(root, 'steps')
    _require({p.name for p in step_dir.iterdir()} == expected_names, 'step inventory differs')
    total, previous = 0, None
    for index in range(count):
        _require(status == 'unavailable', 'steps continue after a terminal proof')
        name = f'steps/{index:04d}.json'
        data = read_bytes_bounded(_contained(root, name), max_bytes=run_owner.STEP_BYTES-total)
        total += len(data)
        row = _decode(data)
        _shape(row, ('schema', 'index', 'time', 'endpoint_time', 'source', 'previous_complete_step',
                      'certificate', 'decision'), 'step record')
        _require(row['schema'] == 'tnfr.formed-two-port-actual-inclusion-step.v1'
                 and type(row['index']) is int and row['index'] == index
                 and exact_record(row['time']) == time
                 and exact_record(row['endpoint_time']) == time+run_owner.STEP_SIZE
                 and row['source'] == association and row['previous_complete_step'] == previous,
                 'step identity or source chain differs')
        center, radius = audit_inclusion_certificate(row['certificate'], model, center, radius, time)
        time += run_owner.STEP_SIZE
        status = audit_terminal_record(row['decision'], radius, time, target_lower)
        last_decision = row['decision']
        previous = {'path': name, 'bytes': len(data), 'sha256': sha256_bytes(data)}
    _require(status in ('certified_capture', 'certified_nonarrival')
             and terminal['status'] == status and terminal['source'] == association
             and terminal['last_complete_step'] == previous
             and exact_record(terminal['endpoint_time']) == time
             and type(terminal['step_evidence_reserved_bytes']) is int
             and terminal['step_evidence_reserved_bytes'] == total
             and terminal['conditional_on_successful_post_write_supervisor_return'] is True,
             'successful terminal differs from complete chain')
    _require(audit_terminal_record(terminal['decision'], radius, time, target_lower) == status,
             'final decision differs')
    _require(_json_bytes(terminal['decision']) == _json_bytes(last_decision),
             'final decision primitives differ from last checkpoint')
    return {
        'passed': True, 'status': status, 'verified_steps': count,
        'endpoint_time': time, 'step_bytes': total,
        'mathematical_primitive_evaluations_reexecuted': False,
        'scientific_response_reexecuted': False,
        'auditor_runtime_association': audit_runtime,
        'scope': ('Stored exact arithmetic, source associations and PSD slack verified.',
                  'Picard/jet/cosine/field/potential and retained-target primitive bounds retain reviewed producer premises.',
                  'Runtime association uses the same declared auditor environment and the reviewed direct-base-image bootstrap; it is not binary authentication.',
                  'Post-write supervisor success required; receipts do not authenticate chronology.'),
    }
