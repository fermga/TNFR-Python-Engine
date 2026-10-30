"""Preserve the failed bookkeeping attempt and prepare its narrow correction."""
from pathlib import Path
import subprocess
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
baseline = json_loads((OUT / 'baseline.json').read_bytes())
before = baseline['dirty_paths_before'].splitlines()
after = subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=ROOT, text=True, encoding='utf-8').splitlines()
added, removed = sorted(set(after) - set(before)), sorted(set(before) - set(after))
assert added == [' M theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md'] and not removed
assert not any((OUT / n).exists() for n in ('source-manifest.json', 'final-maintained-source.zip', 'final-state.json'))
write_json_once(OUT / 'finalization-attempt-v1.json', {
    'passed': False, 'exit_code': 1, 'script': receipt(OUT / 'finalize.py'),
    'failed_assertion': "dirty == baseline['dirty_paths_before']", 'line': 49,
    'added_status_lines': added, 'removed_status_lines': removed,
    'source_files_checked_before_failure': 4635, 'only_declared_two_owners_changed': True,
    'final_outputs_written': False,
    'cause': 'Bookkeeping incorrectly required unchanged dirty-path inventory even though the declared compatibility owner was initially clean.',
    'boundary': 'No scientific execution failed or is repeated. Preserve original finalizer; narrow v2 permits exactly the intended new dirty owner.'})
old = (OUT / 'finalize.py').read_bytes()
needle = b"assert dirty == baseline['dirty_paths_before']"
replacement = b"""# The compatibility owner was clean at baseline; its declared edit adds one
# status line. Every previous status and all unrelated bytes remain preserved.
before_status = baseline['dirty_paths_before'].splitlines()
after_status = dirty.splitlines()
assert set(after_status) - set(before_status) == {' M ' + CHAPTER}
assert set(before_status) - set(after_status) == set()
assert [r for r in after_status if r[3:] not in CHANGED] == [r for r in before_status if r[3:] not in CHANGED]"""
if b'\r\n' in old:
    replacement = replacement.replace(b'\n', b'\r\n')
assert old.count(needle) == 1
with (OUT / 'finalize-v2.py').open('xb') as output:
    output.write(old.replace(needle, replacement))
write_json_once(OUT / 'finalization-correction.json', {
    'failure': receipt(OUT / 'finalization-attempt-v1.json'),
    'original': receipt(OUT / 'finalize.py'), 'corrected': receipt(OUT / 'finalize-v2.py'),
    'preparer': receipt(Path(__file__)),
    'scope': 'Only replace dirty-inventory equality with preservation of all old entries and exactly one declared new modified owner.'})
print({'corrected': 'finalize-v2.py', 'preserved_dirty_paths': len(before), 'new_declared_dirty_paths': len(added)})
