"""Freeze final integration controls; no scientific evaluation."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, verify_archive_members, write_json_once

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
baseline=json_loads((OUT/'baseline.json').read_bytes())
names={p.relative_to(ROOT).as_posix() for p in (ROOT/'src/tnfr').rglob('*.py')}
names.update(baseline['archived_paths']+baseline['planned_added_paths'])
names.update(['AGENTS.md','CONTRIBUTING.md','TESTING.md','pyproject.toml','tests/conftest.py',
    'theory/research_prerequisites.json','theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md',
    'theory/nodal/SINE_FORMED_CLASS_CONTACT.md',
    'artifacts/formed-two-port-executable-20261029/admission_controls.py'])
names.update('docs/assets/sine_formed_classes/two-port-compatibility-v1'+s for s in ('.json','.manifest.json','.protocol.json','.source.zip'))
names.update(p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.suffix in ('.py','.md'))
rows=[receipt(ROOT/n) for n in sorted(names)]
assert sum(r['bytes'] for r in rows)<64*1024**2
with ZipFile(OUT/'controls-source.zip','x',ZIP_DEFLATED) as archive:
    for n in sorted(names):archive.write(ROOT/n,n)
verify_archive_members(OUT/'controls-source.zip',{r['path']:r['sha256'] for r in rows},max_bytes=64*1024**2)
write_json_once(OUT/'controls-freeze.json',{'head':baseline['head'],'before_controls':True,
    'reserved_reference_steps_authorized':False,'records':rows,'archive':receipt(OUT/'controls-source.zip')})
print({'frozen_files':len(rows),'expanded_bytes':sum(r['bytes'] for r in rows)})
