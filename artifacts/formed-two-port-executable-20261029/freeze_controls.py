"""Freeze declared non-transit controls without scientific imports/evaluation."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess
from tnfr.research.artifact_io import file_receipt, write_json_once, verify_archive_members
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
baseline = json_loads((OUT / 'baseline.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == baseline['head']
names = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/tnfr').rglob('*.py')}
names.update(baseline['planned_added_paths'])
names.update(baseline['archived_paths'])
names.update([
    'theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md', 'theory/nodal/SINE_FORMED_CLASS_CONTACT.md',
    'theory/research_prerequisites.json', 'AGENTS.md', 'CONTRIBUTING.md', 'TESTING.md',
    'pyproject.toml', 'tests/conftest.py',
])
names.update('docs/assets/sine_formed_classes/two-port-compatibility-v1' + suffix
             for suffix in ('.json', '.manifest.json', '.protocol.json', '.source.zip'))
names.update(p.relative_to(ROOT).as_posix() for p in OUT.glob('*.md'))
names.update(p.relative_to(ROOT).as_posix() for p in OUT.glob('*controls.py'))
names.add((OUT / 'run_controls.py').relative_to(ROOT).as_posix())
records = [receipt(ROOT / n) for n in sorted(names)]
assert sum(r['bytes'] for r in records) < 64 * 1024**2
with ZipFile(OUT / 'controls-source.zip', 'x', ZIP_DEFLATED) as archive:
    for n in sorted(names):
        archive.write(ROOT / n, n)
verify_archive_members(OUT / 'controls-source.zip', {r['path']: r['sha256'] for r in records}, max_bytes=64*1024**2)
write_json_once(OUT / 'controls-freeze.json', {
    'head': baseline['head'], 'before_controls': True,
    'reserved_reference_steps_authorized': False, 'records': records,
    'archive': receipt(OUT / 'controls-source.zip'),
})
print({'frozen_files': len(records), 'expanded_bytes': sum(r['bytes'] for r in records)})
