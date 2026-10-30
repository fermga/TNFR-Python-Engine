"""Check source preservation and close this documentation-only review."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime, timezone
import subprocess
from copy import deepcopy
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, verify_archive_members, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ALLOWED = {'theory/RESEARCH_PROGRAM.md', 'theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md', 'theory/research_prerequisites.json'}

def read(name):
    return json_loads((OUT / name).read_bytes())

def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)

def verify(item):
    current = receipt(ROOT / item['path'])
    assert current['sha256'] == item['sha256'], item['path']
    if 'bytes' in item:
        assert current['bytes'] == item['bytes'], item['path']

def receipts(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        for child in value.values():
            yield from receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from receipts(child)

def outside_regions(data):
    for start, end in (
        (b'## Continuation point:', b'<a id="foundational-pillars"></a>'),
        (b'The [native foundation account]', b'<a id="foundation-refactoring-plan">'),
    ):
        assert data.count(start) == 1
        left = data.index(start)
        right = data.index(end, left)
        data = data[:left] + b'[REVIEWED REGION]\r\n' + data[right:]
    return data

baseline = read('baseline.json')
scope = read('scope.json')
assert set(baseline['allowed_changed_paths']) == ALLOWED
assert set(scope['allowed_maintained_changes']) == ALLOWED
assert scope['new_scientific_evaluations'] == 0
assert scope['new_trajectories'] == scope['historical_producer_replays'] == 0
assert git('rev-parse', 'HEAD').strip() == baseline['head']
assert git('status', '--porcelain=v1', '--untracked-files=all') == baseline['dirty_paths_before']
names = sorted({n for n in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0') if n and (ROOT / n).is_file()})
original = baseline['source_sha256']
assert len(names) == len(original) == 4626 and set(names) == set(original)
current = {n: sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
changed = {n for n in names if current[n] != original[n]}
assert changed == ALLOWED, changed
verify(baseline['prior_final_state'])
prior = json_loads((ROOT / baseline['prior_final_state']['path']).read_bytes())
assert prior['head'] == baseline['head']
for item in prior['final_sources']:
    assert original[item['path']] == item['sha256']
historical = {item['path']: item for item in receipts(prior) if item['path'].startswith('artifacts/')}
for item in historical.values():
    verify(item)
verify_archive_members(OUT / 'before-edit.zip', {n: original[n] for n in ALLOWED})
chapter = 'theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md'
program = 'theory/RESEARCH_PROGRAM.md'
with ZipFile(OUT / 'before-edit.zip') as archive:
    old_chapter = archive.read(chapter)
    old_program = archive.read(program)
assert len(old_chapter) == scope['chapter_original_prefix_bytes'] == 150187
appended = (OUT / 'joint-review-section.md').read_text(encoding='utf-8').replace('\n', '\r\n').encode('utf-8')
assert (ROOT / chapter).read_bytes() == old_chapter + appended
assert outside_regions(old_program) == outside_regions((ROOT / program).read_bytes())
expected_program = old_program
for start, end, draft in (
    (b'## Continuation point:', b'<a id="foundational-pillars"></a>', 'program-introduction.md'),
    (b'The [native foundation account]', b'<a id="foundation-refactoring-plan">', 'program-decision.md'),
):
    left = expected_program.index(start)
    right = expected_program.index(end, left)
    replacement = (OUT / draft).read_text(encoding='utf-8').replace('\n', '\r\n').encode('utf-8')
    expected_program = expected_program[:left] + replacement + b'\r\n' + expected_program[right:]
assert (ROOT / program).read_bytes() == expected_program
assert (ROOT / 'AGENTS.md').read_bytes() == (ROOT / '.github/agents/my-agent.md').read_bytes()
assert (ROOT / program).stat().st_size <= 96 * 1024
assert len((ROOT / chapter).read_bytes().splitlines()) <= 4000
assert (ROOT / program).read_bytes().count(b'<!-- research-activity: foundations -->') == 1
applied = read('applied-decision.json')
for key in ('chapter', 'program'):
    verify(applied[key])
for key in ('drafts', 'independent_reviews'):
    for item in applied[key]:
        verify(item)
for name in ('mathematical-review.json', 'scope-review.json', 'observation-review.json'):
    review = read(name)
    if name == 'mathematical-review.json':
        assert review['decision'] == 'PASS'
        assert review['independence']['distinct_author_and_reviewer'] is True
        assert review['independence']['reviewer_authored_reviewed_drafts'] is False
        assert review['independence']['reviewer_read_final_draft_bytes'] is True
    else:
        assert review['verdict'] == 'PASS' and review['independent'] is True
    assert review['author'] != review['reviewer']
    draft_receipts = review['reviewed_drafts'] if name == 'mathematical-review.json' else review['sources']
    assert {r['path'] for r in draft_receipts} == {r['path'] for r in applied['drafts']}
    for item in draft_receipts:
        verify(item)
registry = json_loads((ROOT / 'theory/research_prerequisites.json').read_bytes())
with ZipFile(OUT / 'before-edit.zip') as archive:
    old_registry = json_loads(archive.read('theory/research_prerequisites.json'))
assert registry['physical_mapping_readiness_review'] is None
assert set(registry['requirements']) == set(old_registry['requirements'])
def unchanged_registry_contract(value):
    normalized = deepcopy(value)
    for requirement in normalized['requirements'].values():
        requirement.pop('independent_review')
        for evidence in requirement['evidence']:
            evidence.pop('sha256')
    return normalized
assert unchanged_registry_contract(registry) == unchanged_registry_contract(old_registry)
for identifier, requirement in registry['requirements'].items():
    assert requirement['status'] == 'resolved'
    verify(requirement['independent_review'])
    review = json_loads((ROOT / requirement['independent_review']['path']).read_bytes())
    assert review['requirement'] == identifier and review['verdict'] == 'approved'
    assert review['independent'] is True and review['author'] != review['reviewer']
    assert requirement['evidence'] == review['evidence']
    assert [(r['path'], r['anchor']) for r in requirement['evidence']] == [(r['path'], r['anchor']) for r in old_registry['requirements'][identifier]['evidence']]
    for item in review['evidence']:
        verify(item)
assert read('prerequisite-validation.json')['passed'] is True
verify(read('prerequisite-validation.json')['registry'])
assert read('documentation-validation.json')['passed'] is True
for item in read('documentation-validation.json')['validated_sources']:
    verify(item)
for name in ('prerequisite-validation.json', 'documentation-validation.json'):
    for check in read(name)['checks']:
        verify(check['log'])
assert [c['exit_code'] for c in read('prerequisite-validation.json')['checks']] == [0, 1]
assert len(read('documentation-validation.json')['checks']) == 5
assert all(c['exit_code'] == 0 for c in read('documentation-validation.json')['checks'])
assert not read('rendered-links.json')['failures']
final_review = read('final-admission-review.json')
assert final_review['verdict'] == 'PASS' and final_review['independent'] is True
assert final_review['author'] != final_review['reviewer']
required_final = ALLOWED | {
    (OUT / name).relative_to(ROOT).as_posix()
    for name in (
        'applied-decision.json', 'mathematical-review.json', 'scope-review.json',
        'observation-review.json', 'prerequisite-validation.json',
        'documentation-validation.json', 'rendered-links.json', 'finalize.py',
        'operator_causality-review.json', 'state_observation-review.json',
        'organizational_identity-review.json', 'research_selection-review.json',
    )
}
assert {r['path'] for r in final_review['reviewed_files']} == required_final
assert len(final_review['reviewed_files']) == len(required_final)
for item in final_review['reviewed_files']:
    verify(item)
with ZipFile(OUT / 'final-maintained-source.zip', 'x', ZIP_DEFLATED) as archive:
    for name in sorted(ALLOWED):
        archive.write(ROOT / name, arcname=name)
verify_archive_members(OUT / 'final-maintained-source.zip', {n: current[n] for n in ALLOWED})
records = [
    'scope.json', 'baseline.json', 'before-edit.zip', 'applied-decision.json',
    'mathematical-review.json', 'scope-review.json', 'observation-review.json',
    'prerequisite-validation.json', 'documentation-validation.json',
    'rendered-links.json', 'final-admission-review.json', 'finalize.py',
    'final-maintained-source.zip',
]
write_json_once(OUT / 'final-state.json', {
    'passed': True,
    'date_utc': datetime.now(timezone.utc).isoformat(),
    'head': baseline['head'],
    'baseline_files': len(original),
    'unchanged_files': len(names) - len(changed),
    'changed_paths': sorted(changed),
    'added_paths': [], 'removed_paths': [],
    'dirty_paths_preserved': True,
    'dirty_path_count': len(baseline['dirty_paths_before'].splitlines()),
    'source_sha256': current,
    'prior_final_state': baseline['prior_final_state'],
    'chapter_original_prefix_preserved_bytes': len(old_chapter),
    'program_outside_declared_regions_preserved': True,
    'program_regions_match_reviewed_drafts': True,
    'historical_artifact_receipts_verified': len(historical),
    'records': [receipt(OUT / name) for name in records],
    'prerequisite_scope': 'Original prepared single-bridge scope; no enlargement.',
    'physical_mapping_readiness_review': None,
    'physical_mapping_authorized': False,
    'new_scientific_evaluations': 0,
    'new_trajectories': 0,
    'historical_producer_replays': 0,
    'next_dependency': 'Finite receiver-form observation admission for the one-bridge JO matched-capacity response.',
})
print({'passed': True, 'maintained_files': len(names), 'unchanged': len(names)-len(changed), 'changed': sorted(changed), 'new_scientific_evaluations': 0})
