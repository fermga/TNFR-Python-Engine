"""Manufactured documentary audits; real scientific evaluation is forbidden."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as Q
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric_inclusion import validated_metric_inclusion_step
from tnfr.research import formed_two_port_arrival as model_owner
from tnfr.research import formed_two_port_arrival_audit as audit
from tnfr.research import formed_two_port_arrival_run as run
from tnfr.research import formed_two_port_reference as reference
from tnfr.research.artifact_io import decode_exact_tree, file_receipt, sha256_bytes
from tnfr.utils.io import json_loads


def decoded(value):
    return decode_exact_tree(json_loads(run._json_bytes(value)))


@pytest.fixture(autouse=True)
def forbid_science(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('documentary audit invoked scientific evaluation')
    monkeypatch.setattr(reference.FormedTwoPortReference, 'field', forbidden)
    monkeypatch.setattr(reference.FormedTwoPortReference, 'angles', forbidden)
    monkeypatch.setattr(reference, 'admit_formed_two_port_target', forbidden)
    monkeypatch.setattr(reference, 'admit_formed_two_port_acute_exit_prefix', forbidden)
    monkeypatch.setattr(model_owner, 'assess_formed_two_port_arrival_terminal', forbidden)


@pytest.fixture(scope='module')
def synthetic_certificate():
    # Constant zero drift and zero stiffness are a manufactured inclusion,
    # not the scientific sine field at any preparation.
    model = model_owner.build_formed_two_port_arrival_model()
    center = (Q(0),)*17
    radius = Q(1, 2**30)+Q(1, 45000)
    def certify(tube):
        return model_owner._graph_growth(model.chart, model.reference.oriented_edges,
                                         (Q(0),)*20, tube, run.RATE_BOUNDS,
                                         run.RATE_BISECTIONS)
    result, failed, reason = validated_metric_inclusion_step(
        center, radius, model.metric, run.STEP_SIZE,
        lambda state: tuple(0*x for x in state), lambda _: (Q(1),),
        forcing_bound=Q(1, 7000), forcing_refinement=lambda _: Q(1, 7000),
        growth_certifier=certify, order=run.ORDER, time=1,
    )
    assert result is not None and failed is reason is None
    return model, center, radius, decoded(asdict(result))


def test_arithmetic_audit_rebuilds_synthetic_full_certificate(synthetic_certificate):
    model, center, radius, cert = synthetic_certificate
    observed = audit.audit_inclusion_certificate(cert, model, center, radius, Q(1))
    assert observed == (cert['endpoint_center'], cert['endpoint_radius'])


@pytest.mark.parametrize('change', ('carry', 'gain', 'forcing', 'slack', 'tube', 'float', 'schema'))
def test_corrupted_inclusion_records_are_refused(synthetic_certificate, change):
    model, center, radius, original = synthetic_certificate
    cert = deepcopy(original)
    if change == 'carry':
        cert['nominal_step']['initial_radius'] += 1
    elif change == 'gain':
        cert['forcing_gain_upper_bound'] = 0
    elif change == 'forcing':
        cert['effective_forcing_bound'] = -1
    elif change == 'slack':
        cert['external_growth_certificate']['slack_matrix'] = ((Q(0),)*17,)*17
    elif change == 'tube':
        cert['growth_proof_tube'] = ({'lo': Q(0), 'hi': Q(0)},)*17
    elif change == 'float':
        cert['nominal_step']['local_metric_error_upper_bound'] = 0.0
    else:
        cert['unexpected'] = 1
    with pytest.raises((ValueError, TypeError)):
        audit.audit_inclusion_certificate(cert, model, center, radius, Q(1))


def test_terminal_record_is_rebuilt_and_boolean_math_rejected():
    decision = decoded(asdict(model_owner._terminal_arithmetic(2, 1, 0, 0, 1, 2)))
    assert audit.audit_terminal_record(decision, Q(0), Q(2), Q(2)) == 'certified_nonarrival'
    decision['form_storage_upper_bound'] = False
    with pytest.raises(ValueError):
        audit.audit_terminal_record(decision, Q(0), Q(2), Q(2))


def synthetic_attempt(tmp_path, synthetic_certificate, monkeypatch):
    model, center, radius, certificate = synthetic_certificate
    old_radius = Q(1, 2**30)
    names = run._OWNERS | run._PREFIX_FILES | run._TARGET_FILES | {'worker.py', 'src/tnfr/synthetic.py'}
    contents = {name: b'manufactured proof fixture only\n' for name in names}
    contents[f'{run._PREFIX}/attempt/steps/0003.json'] = run._json_bytes({
        'certificate': {'endpoint_center': (Q(0),)*18, 'endpoint_radius': old_radius}})
    rows = tuple({'path': name, 'bytes': len(data), 'sha256': sha256_bytes(data)}
                 for name, data in sorted(contents.items()))
    source_rows = {x['path']: x for x in rows}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    (tmp_path/'source.zip').write_bytes(stream.getvalue())
    archive_receipt = file_receipt(tmp_path/'source.zip', label='source.zip')
    reviews = []
    for index in range(2):
        path = tmp_path/f'review-{index:02d}.md'
        path.write_bytes(b'Synthetic review association, not scientific approval.\n')
        reviews.append(file_receipt(path, label=f'original-review-{index}.md'))
    (tmp_path/'controls.json').write_bytes(run._json_bytes({'passed': True, 'reserved_actual_steps': 0}))
    controls = file_receipt(tmp_path/'controls.json', label='original-controls.json')
    runtime = {'python_executable': str(tmp_path/'python.exe'),
               'python_prefix': str(tmp_path/'environment'), 'python_version': 'synthetic',
               'package_root': str(tmp_path/'src/tnfr')}
    base_image = str(tmp_path/'base-python/python.exe')
    monkeypatch.setattr(audit, '_current_audit_runtime',
                        lambda: {**runtime, 'base_executable': base_image})
    protocol = {'schema': run.SCHEMA, 'policy': run.declared_policy(), 'head': 'a'*40,
                'runtime': runtime, 'sources': rows,
                'archive': {**archive_receipt, 'path': 'original-source.zip'},
                'reviews': reviews, 'controls': controls,
                'worker_entry': source_rows['worker.py']}
    (tmp_path/'protocol.json').write_bytes(run._json_bytes(protocol))
    association = {'protocol': file_receipt(tmp_path/'protocol.json', label='protocol.json'),
                   'archive': archive_receipt, 'head': protocol['head'],
                   'worker_entry': protocol['worker_entry'],
                   'target_evidence': tuple(source_rows[x] for x in sorted(run._TARGET_FILES)),
                   'prefix_evidence': tuple(source_rows[x] for x in sorted(run._PREFIX_FILES))}
    prefix_receipts = tuple({**source_rows[x], 'path': x.removeprefix(run._PREFIX+'/')}
                            for x in sorted(run._PREFIX_FILES))
    source = model_owner.ArrivalSource(Q(1), center, radius, old_radius, Q(1, 45000),
                                       prefix_receipts, Q(0), Q(1, 2**26)-old_radius)
    target_receipts = association['target_evidence']
    retained = reference.FormedTwoPortTarget((I(0),)*18, (I(0),)*18, ((Q(0),)*3,)*18,
                                             Q(1), target_receipts, 'synthetic', 'synthetic')
    target = model_owner.ArrivalTarget(retained, I(2), (I(0),)*18, 'synthetic', target_receipts)
    initial = model_owner._terminal_arithmetic(1, 1, radius, 0, 3, 2)
    final = model_owner._terminal_arithmetic(Q(17, 16), 1, certificate['endpoint_radius'], 0, 1, 2)
    assert initial.status == 'unavailable' and final.status == 'certified_nonarrival'
    admission = {'source': association, 'model': asdict(model), 'actual_source': asdict(source),
                  'target': asdict(target), 'initial_decision': asdict(initial),
                  'full_law_and_original_source_preserved': True}
    (tmp_path/'admission.json').write_bytes(run._json_bytes(admission))
    (tmp_path/'steps').mkdir()
    record = {'schema': 'tnfr.formed-two-port-actual-inclusion-step.v1', 'index': 0,
               'time': Q(1), 'endpoint_time': Q(17, 16), 'source': association,
               'previous_complete_step': None, 'certificate': certificate, 'decision': asdict(final)}
    step_path = tmp_path/'steps/0000.json'
    step_path.write_bytes(run._json_bytes(record))
    step_receipt = file_receipt(step_path, label='steps/0000.json')
    terminal = {'status': final.status, 'completed_steps': 1, 'endpoint_time': Q(17, 16),
                 'source': association, 'last_complete_step': step_receipt, 'decision': asdict(final),
                 'step_evidence_reserved_bytes': step_receipt['bytes'],
                 'conditional_on_successful_post_write_supervisor_return': True,
                 'boundary': 'Manufactured audit fixture only.'}
    (tmp_path/'worker-result.json').write_bytes(run._json_bytes(terminal))
    attempt = {'script': str(tmp_path/'worker.py'),
               'arguments': ('--root', str(tmp_path), '--protocol', str(tmp_path/'protocol.json'),
                             '--output', str(tmp_path)), 'cwd': str(tmp_path),
               'interpreter': runtime['python_executable'], 'prefix': runtime['python_prefix'],
               'wall_seconds': 3600, 'closing_reserve_seconds': 10.0,
               'worker_seconds_budget': 3590.0, 'process_memory_bytes': 1024**3,
               'memory_metric': 'Windows committed virtual memory per process and job',
               'log_limit_bytes': 1024**2, 'one_worker': True}
    (tmp_path/'attempt.json').write_bytes(run._json_bytes(attempt))
    (tmp_path/'worker-start.json').write_bytes(run._json_bytes({
        'schema': run.SCHEMA, 'job': {'synthetic': True}, 'policy': run.declared_policy()}))
    (tmp_path/'worker.log').write_bytes(b'')
    snapshot = dict(record_stage='pre_write_snapshot_not_final_attempt_verdict', passed=True,
                    status='success', exit_code=0, job_assigned_before_resume=True, resumed=True,
                    worker_stopped_observed=True, cleanup_complete=True,
                    terminal_write_deadline_verified=False, watchdog_fired=False,
                    watchdog_error=None, total_budget_exhausted=False,
                    supervisor_elapsed_seconds=1.0, wall_seconds_limit=3600,
                    image=base_image,
                    process_memory_limit_bytes=1024**3,
                    attempt=file_receipt(tmp_path/'attempt.json', label='attempt.json'),
                    log=file_receipt(tmp_path/'worker.log', label='worker.log'))
    (tmp_path/'execution.json').write_bytes(run._json_bytes(snapshot))
    returned = {**snapshot, 'record_stage': 'post_write_result_required_for_attempt_admission',
                'terminal_write_deadline_verified': True, 'supervisor_elapsed_seconds': 2.0}
    return returned


def test_complete_manufactured_chain_is_audited_without_scientific_replay(tmp_path, synthetic_certificate, monkeypatch):
    returned = synthetic_attempt(tmp_path, synthetic_certificate, monkeypatch)
    report = audit.audit_formed_two_port_arrival_attempt(tmp_path, post_write_supervisor_result=returned)
    assert report['passed'] and report['verified_steps'] == 1
    assert report['status'] == 'certified_nonarrival'
    assert report['scientific_response_reexecuted'] is False
    assert report['mathematical_primitive_evaluations_reexecuted'] is False
    runtime = report['auditor_runtime_association']
    assert runtime['python_executable'] != runtime['base_executable']
    assert runtime['base_executable'] == returned['image']


@pytest.mark.parametrize('change', ('prewrite', 'timeout', 'final_primitive', 'chain', 'extra_step', 'model_bool', 'empty_target', 'runtime'))
def test_final_or_documentary_tampering_cannot_certify_success(tmp_path, synthetic_certificate, change, monkeypatch):
    returned = synthetic_attempt(tmp_path, synthetic_certificate, monkeypatch)
    if change == 'prewrite':
        returned['record_stage'] = 'pre_write_snapshot_not_final_attempt_verdict'
    elif change == 'timeout':
        returned['supervisor_elapsed_seconds'] = 3601.0
    elif change == 'extra_step':
        (tmp_path/'steps/0001.json').write_bytes(b'{}')
    elif change == 'model_bool':
        path = tmp_path/'admission.json'
        record = decode_exact_tree(json_loads(path.read_bytes()))
        record['model']['reference']['initial_center'] = (False,)*18
        path.write_bytes(run._json_bytes(record))
    elif change == 'empty_target':
        path = tmp_path/'admission.json'
        record = decode_exact_tree(json_loads(path.read_bytes()))
        record['target']['retained_target']['target_phases'] = ()
        path.write_bytes(run._json_bytes(record))
    elif change == 'runtime':
        path = tmp_path/'attempt.json'
        record = decode_exact_tree(json_loads(path.read_bytes()))
        record['interpreter'] = str(tmp_path/'different.exe')
        path.write_bytes(run._json_bytes(record))
        changed_receipt = file_receipt(path, label='attempt.json')
        returned['attempt'] = changed_receipt
        snapshot_path = tmp_path/'execution.json'
        snapshot = decode_exact_tree(json_loads(snapshot_path.read_bytes()))
        snapshot['attempt'] = changed_receipt
        snapshot_path.write_bytes(run._json_bytes(snapshot))
    else:
        path = tmp_path/'worker-result.json'
        record = decode_exact_tree(json_loads(path.read_bytes()))
        if change == 'chain':
            record['last_complete_step']['sha256'] = '0'*64
        else:
            cert_radius = synthetic_certificate[3]['endpoint_radius']
            record['decision'] = asdict(model_owner._terminal_arithmetic(Q(17, 16), 2, cert_radius, 0, 1, 2))
        path.write_bytes(run._json_bytes(record))
    with pytest.raises((ValueError, TypeError)):
        audit.audit_formed_two_port_arrival_attempt(tmp_path, post_write_supervisor_result=returned)


@pytest.mark.parametrize('distinct_base', (False, True))
def test_runtime_association_accepts_checked_base_or_venv_launch(tmp_path, monkeypatch, distinct_base):
    frozen = {'python_executable': str(tmp_path/'venv/python.exe'),
              'python_prefix': str(tmp_path/'venv'), 'python_version': 'synthetic',
              'package_root': str(tmp_path/'src/tnfr')}
    image = str(tmp_path/'base/python.exe') if distinct_base else frozen['python_executable']
    current = {**frozen, 'base_executable': image}
    monkeypatch.setattr(audit, '_current_audit_runtime', lambda: current)
    attempt = {'interpreter': frozen['python_executable'], 'prefix': frozen['python_prefix']}
    assert audit._admit_runtime_association(frozen, attempt, image) == current


@pytest.mark.parametrize('change', ('base_image', 'python_executable', 'python_prefix',
                                   'python_version', 'package_root',
                                   'attempt_interpreter', 'attempt_prefix'))
def test_runtime_association_rejects_a_different_environment(tmp_path, monkeypatch, change):
    frozen = {'python_executable': str(tmp_path/'venv/python.exe'),
              'python_prefix': str(tmp_path/'venv'), 'python_version': 'synthetic',
              'package_root': str(tmp_path/'src/tnfr')}
    image = str(tmp_path/'base/python.exe')
    current = {**frozen, 'base_executable': image}
    attempt = {'interpreter': frozen['python_executable'], 'prefix': frozen['python_prefix']}
    if change == 'base_image':
        image = str(tmp_path/'unrelated/python.exe')
    elif change.startswith('attempt_'):
        attempt[change.removeprefix('attempt_')] = str(tmp_path/'unrelated')
    else:
        current[change] = 'different version' if change == 'python_version' else str(tmp_path/'unrelated')
    monkeypatch.setattr(audit, '_current_audit_runtime', lambda: current)
    with pytest.raises(ValueError):
        audit._admit_runtime_association(frozen, attempt, image)


@pytest.mark.parametrize('has_base', (False, True))
def test_runtime_snapshot_uses_base_executable_with_native_fallback(tmp_path, monkeypatch, has_base):
    executable, base = tmp_path/'venv/python.exe', tmp_path/'base/python.exe'
    runtime = SimpleNamespace(executable=str(executable), prefix=str(tmp_path/'venv'),
                              version='synthetic')
    if has_base:
        runtime._base_executable = str(base)
    monkeypatch.setattr(audit, 'sys', runtime)
    observed = audit._current_audit_runtime()
    assert observed['python_executable'] == str(executable.resolve())
    assert observed['base_executable'] == str((base if has_base else executable).resolve())
    assert observed['python_prefix'] == str((tmp_path/'venv').resolve())
    assert observed['python_version'] == 'synthetic'
    assert observed['package_root'] == str(Path(audit.__file__).resolve().parents[1])
