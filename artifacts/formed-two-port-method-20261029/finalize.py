"""Close exact method controls and documentary consolidation without transit."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, verify_archive_members, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHAPTER = 'theory/nodal/SINE_TWO_PORT_CAPTURE.md'
PROGRAM = 'theory/RESEARCH_PROGRAM.md'
CHANGED = {CHAPTER, PROGRAM}
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
read = lambda p: json_loads(p.read_bytes())
git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True, encoding='utf-8')


def verify(row):
    assert receipt(ROOT / row['path']) == row, row['path']


baseline = read(OUT / 'baseline.json')
applied = read(OUT / 'applied-integration.json')
original = baseline['source_sha256']
for key in ('prior_final_state', 'prior_source_manifest', 'archive'):
    verify(baseline[key])
prior = read(ROOT / baseline['prior_final_state']['path'])
assert prior['passed'] and prior['source_sha256'] == original
assert len(prior['records']) == baseline['prior_records_verified'] == 50
for row in prior['records']:
    verify(row)
verify(prior['prior_final_state'])
older = read(ROOT / prior['prior_final_state']['path'])
for row in older['records']:
    verify(row)
for row in baseline['books']:
    verify(row)
assert len(baseline['books']) == 7
assert {p.relative_to(ROOT).as_posix() for p in (ROOT / 'bibliografia').iterdir() if p.is_file()} == {r['path'] for r in baseline['books']}
verify_archive_members(OUT / 'before-edit.zip', {n: original[n] for n in CHANGED})
assert git('rev-parse', 'HEAD').strip() == baseline['head']
names = {n for n in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0') if n and (ROOT / n).is_file()}
assert names == set(original)
current = {n: sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(names)}
assert {n for n in original if original[n] != current[n]} == CHANGED
dirty = git('status', '--porcelain=v1', '--untracked-files=all')
before_status, after_status = baseline['dirty_paths_before'].splitlines(), dirty.splitlines()
# This declared owner was clean; retain every previously dirty status verbatim.
assert set(after_status) - set(before_status) == {' M ' + CHAPTER}
assert set(before_status) - set(after_status) == set()
assert [r for r in after_status if r[3:] not in CHANGED] == [r for r in before_status if r[3:] not in CHANGED]
assert (ROOT / 'AGENTS.md').read_bytes() == (ROOT / '.github/agents/my-agent.md').read_bytes()
for row in applied['associations']:
    verify(row)
for item in applied['files']:
    verify(item['proposal'])
    verify(item['maintained'])
    assert (ROOT / item['proposal']['path']).read_bytes() == (ROOT / item['maintained']['path']).read_bytes()
assert applied['new_analytic_method']
for key in ('actual_source_transits', 'reference_steps', 'root_solves', 'historical_producer_replays'):
    assert applied[key] == 0
with ZipFile(OUT / 'before-edit.zip') as archive:
    old_chapter, old_program = archive.read(CHAPTER), archive.read(PROGRAM)
assert (ROOT / CHAPTER).read_bytes() == old_chapter + (OUT / 'chapter-append-v3.md').read_bytes()

# Reconstruct only the declared program changes, without rerunning writers.
encode = lambda text: text.replace('\n', '\r\n' if b'\r\n' in old_program else '\n').encode()
expected_program = old_program
for edit in read(OUT / 'drafts.json')['replacements']:
    a, b = encode(edit['old']), encode(edit['new'])
    assert expected_program.count(a) == 1
    expected_program = expected_program.replace(a, b)
start, end = b'### Sole active question and first obligation', b'### Source context and available methods'
assert expected_program.count(start) == expected_program.count(end) == 1
expected_program = expected_program[:expected_program.index(start)] + encode((OUT / 'program-current.md').read_text(encoding='utf-8')) + expected_program[expected_program.index(end):]
new_program = (ROOT / PROGRAM).read_bytes()
assert expected_program == new_program
freeze = read(OUT / 'source-freeze.json')
verify(freeze['archive'])
verify(freeze['script'])
verify_archive_members(ROOT / freeze['archive']['path'], {r['path']: r['sha256'] for r in freeze['records']})
assert freeze['before_exact_controls'] and freeze['actual_source_transit_authorized'] is False
assert freeze['head'] == baseline['head']
for row in freeze['records']:
    if row['path'] in CHANGED:
        assert row['sha256'] == original[row['path']]
    else:
        verify(row)
controls = read(OUT / 'method-controls.json')
execution = read(OUT / 'exact-execution.json')
assert controls['passed'] and len(controls['checks']) == 30
for key in ('source', 'protocol', 'review'):
    verify(controls[key])
for key in ('actual_source_transits', 'reference_steps', 'root_solves', 'target_sign_assessments', 'response_assessments', 'historical_producer_replays'):
    assert controls[key] == 0
assert execution['passed'] and execution['exit_code'] == 0 and execution['timed_out'] is False
assert execution['timeout_seconds'] == 60
assert execution['command'][1] == '-E' and '-O' not in execution['command']
assert Path(execution['command'][2]).resolve() == (OUT / 'check_method.py').resolve()
for key in ('source_freeze', 'log', 'wrapper'):
    verify(execution[key])
guards = read(OUT / 'prerequisite-validation.json')
docs = read(OUT / 'documentation-validation.json')
rendered = read(OUT / 'rendered-links.json')
assert guards['passed'] and [r['exit_code'] for r in guards['checks']] == [0, 1]
assert guards['physical_mapping_readiness_review'] is None and guards['physical_mapping_authorized'] is False
verify(guards['registry'])
assert 'missing final readiness review' in (ROOT / guards['checks'][1]['log']['path']).read_text(encoding='utf-8')
assert docs['passed'] and len(docs['checks']) == 5 and all(r['exit_code'] == 0 for r in docs['checks'])
for record in (guards, docs):
    for row in record['checks']:
        verify(row['log'])
for row in docs['validated_sources']:
    verify(row)
assert {r['path'] for r in docs['validated_sources']} == CHANGED
assert docs['new_scientific_evaluations_by_documentation_checks'] == 0
assert rendered['failures'] == []
assert all(len((ROOT / n).read_bytes().splitlines()) <= 4000 for n in CHANGED)
assert len(new_program) <= 96 * 1024
assert not list(OUT.rglob('*.png')) and not list(OUT.rglob('*.pdf'))

summary = {'head': baseline['head'], 'source_sha256': current, 'baseline_files': len(original),
           'maintained_files': len(current), 'unchanged_files': len(original) - 2,
           'changed_paths': sorted(CHANGED), 'agent_changed_paths': sorted(CHANGED),
           'added_paths': [], 'removed_paths': []}
write_json_once(OUT / 'source-manifest.json', summary)
with ZipFile(OUT / 'final-maintained-source.zip', 'x', ZIP_DEFLATED) as archive:
    for n in sorted(CHANGED):
        archive.write(ROOT / n, n)
verify_archive_members(OUT / 'final-maintained-source.zip', {n: current[n] for n in CHANGED})
records = [receipt(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'final-state.json']
write_json_once(OUT / 'final-state.json', {
    'passed': True, 'date_utc': datetime.now(timezone.utc).isoformat(), **summary,
    'prior_final_state': baseline['prior_final_state'], 'prior_source_manifest': baseline['prior_source_manifest'],
    'prior_records_preserved': len(prior['records']), 'earlier_records_preserved': len(older['records']),
    'dirty_path_count_before': len(before_status), 'dirty_path_count': len(after_status),
    'new_declared_dirty_paths': [CHAPTER], 'books_preserved_in_place': baseline['books'],
    'new_analytic_method': True, 'exact_method_controls': len(controls['checks']),
    'actual_source_transits': 0, 'reference_steps': 0, 'root_solves': 0,
    'target_sign_assessments': 0, 'response_assessments': 0, 'historical_producer_replays': 0,
    'physical_mapping_readiness_review': None, 'physical_mapping_authorized': False,
    'exact_reviewed_drafts_applied': True, 'next_question_updated': True,
    'next_question': applied['next_question'], 'records': records,
    'boundary': 'Mathematical selection of the actual-source eighteen-phase reference, full-state comparison/tail and prospective resource policy. Future adapter/supervisor implementation, review and producing-source freeze are required before the single reserved transit; actual entry remains unproved.'})
print({'passed': True, 'files': len(current), 'unchanged': len(original) - 2,
       'edited': sorted(CHANGED), 'exact_controls': len(controls['checks']), 'reference_steps': 0,
       'prior_records': len(prior['records']), 'earlier_records': len(older['records']), 'records': len(records)})
