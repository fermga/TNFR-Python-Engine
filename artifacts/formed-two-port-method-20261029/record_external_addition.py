"""Preserve a concurrent unread PDF addition without attributing it to this work."""
from pathlib import Path
from hashlib import sha256
import subprocess
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
baseline = json_loads((OUT / 'baseline.json').read_bytes())
name = 'bibliografia/TNFR.pdf'
entry = receipt(ROOT / name)
assert entry == {'path': name, 'bytes': 2803734, 'sha256': '5ba0f4a2da2d01e550e7004c3c09c6927e6620826052df84cd28babbcbd34fc3'}
git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True, encoding='utf-8')
names = {n for n in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0') if n and (ROOT / n).is_file()}
original = baseline['source_sha256']
assert names - set(original) == {name} and not set(original) - names
changed = {n for n in original if sha256((ROOT / n).read_bytes()).hexdigest() != original[n]}
assert changed == {'theory/RESEARCH_PROGRAM.md', 'theory/nodal/SINE_TWO_PORT_CAPTURE.md'}
for book in baseline['books']:
    assert receipt(ROOT / book['path']) == book
before, after = baseline['dirty_paths_before'].splitlines(), git('status', '--porcelain=v1', '--untracked-files=all').splitlines()
assert set(after) - set(before) == {' M theory/nodal/SINE_TWO_PORT_CAPTURE.md', '?? ' + name}
assert not set(before) - set(after)
assert not any((OUT / n).exists() for n in ('source-manifest.json', 'final-maintained-source.zip', 'final-state.json'))
write_json_once(OUT / 'external-addition.json', {
    'files': [entry], 'origin': 'Concurrent filesystem addition; creator not inferred. Not created or edited by this task.',
    'content_read_or_used': False, 'selected_bibliography_unchanged': baseline['books'],
    'existing_source_changes': sorted(changed), 'new_source_paths': [name],
    'dirty_count_before': len(before), 'dirty_count_after': len(after),
    'scientific_dependency_changed': False,
    'boundary': 'Preserve the new unread historical manuscript. Its presence supplies no premise or additional research result.'})
write_json_once(OUT / 'finalization-attempt-v1.json', {
    'passed': False, 'exit_code': 1, 'script': receipt(OUT / 'finalize.py'),
    'failed_line': 41, 'failed_condition': 'Bibliography inventory equals the seven baseline files',
    'cause': 'A new TNFR.pdf appeared after the baseline/source freeze. All seven original files retain their bytes.',
    'final_outputs_written': False, 'external_addition': receipt(OUT / 'external-addition.json'),
    'scientific_controls_repeated': False})
data = (OUT / 'finalize.py').read_bytes()
changes = [
    ("assert {p.relative_to(ROOT).as_posix() for p in (ROOT / 'bibliografia').iterdir() if p.is_file()} == {r['path'] for r in baseline['books']}",
     "addition = read(OUT / 'external-addition.json')\nfor row in addition['files']:\n    verify(row)\nEXTERNAL = {r['path'] for r in addition['files']}\nassert EXTERNAL == {'bibliografia/TNFR.pdf'}\nassert addition['content_read_or_used'] is False and addition['scientific_dependency_changed'] is False\nassert {p.relative_to(ROOT).as_posix() for p in (ROOT / 'bibliografia').iterdir() if p.is_file()} == {r['path'] for r in baseline['books']} | EXTERNAL"),
    ("assert names == set(original)", "assert names == set(original) | EXTERNAL"),
    ("assert set(after_status) - set(before_status) == {' M ' + CHAPTER}",
     "assert set(after_status) - set(before_status) == {' M ' + CHAPTER} | {'?? ' + n for n in EXTERNAL}"),
    ("assert [r for r in after_status if r[3:] not in CHANGED] == [r for r in before_status if r[3:] not in CHANGED]",
     "assert [r for r in after_status if r[3:] not in CHANGED | EXTERNAL] == [r for r in before_status if r[3:] not in CHANGED | EXTERNAL]"),
    ("'added_paths': [], 'removed_paths': []", "'added_paths': sorted(EXTERNAL), 'externally_added_paths': sorted(EXTERNAL), 'agent_added_paths': [], 'removed_paths': []"),
    ("'new_declared_dirty_paths': [CHAPTER], 'books_preserved_in_place': baseline['books'],",
     "'new_declared_dirty_paths': [CHAPTER], 'external_new_dirty_paths': sorted(EXTERNAL),\n    'books_preserved_in_place': baseline['books'] + addition['files'],\n    'selected_book_sources': baseline['books'], 'unreviewed_external_addition': receipt(OUT / 'external-addition.json'),")]
for old, new in changes:
    a, b = old.encode(), new.replace('\n', '\r\n' if b'\r\n' in data else '\n').encode()
    assert data.count(a) == 1, old
    data = data.replace(a, b)
with (OUT / 'finalize-v2.py').open('xb') as output:
    output.write(data)
write_json_once(OUT / 'finalization-correction.json', {
    'original': receipt(OUT / 'finalize.py'), 'corrected': receipt(OUT / 'finalize-v2.py'),
    'failure': receipt(OUT / 'finalization-attempt-v1.json'), 'addition': receipt(OUT / 'external-addition.json'),
    'preparer': receipt(Path(__file__)),
    'scope': 'Preserve exactly one concurrent unread external file, all original source bytes outside the two approved owners, and all original dirty states. No scientific or documentation check is repeated.'})
print({'preserved_external_addition': entry, 'corrected_finalizer': receipt(OUT / 'finalize-v2.py')})
