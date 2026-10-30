"""Freeze exact producing source/protocol after reviewed integration controls."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess
import sys
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, verify_archive_members, write_json_once
from tnfr.research.formed_two_port_run import declared_policy, PROTOCOL_SCHEMA

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
baseline=json_loads((OUT/'baseline.json').read_bytes())
controls=json_loads((OUT/'controls-result.json').read_bytes())
assert controls['passed'] and controls['reserved_reference_steps']==0
assert controls['exit_code']==0 and controls['timed_out'] is False
assert controls['counts']['tests']>0 and all(controls['counts'][k]==0 for k in ('failures','errors','skipped'))
for key in ('freeze','log','junit'):
    row=controls[key]
    assert receipt(ROOT/row['path'])==row,row['path']
assert controls['freeze']==receipt(OUT/'controls-freeze.json')
assert not(OUT/'attempt').exists() and not(OUT/'caller-result.json').exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline['head']
control_freeze=json_loads((OUT/'controls-freeze.json').read_bytes())
assert receipt(ROOT/control_freeze['archive']['path'])==control_freeze['archive']
for row in control_freeze['records']:
    assert receipt(ROOT/row['path'])==row,row['path']
names={p.relative_to(ROOT).as_posix() for p in (ROOT/'src/tnfr').rglob('*.py')}
assert names=={row['path'] for row in control_freeze['records'] if row['path'].startswith('src/tnfr/') and row['path'].endswith('.py')}
names.update(baseline['archived_paths']+baseline['planned_added_paths'])
names.update(['AGENTS.md','pyproject.toml','.venv313/pyvenv.cfg','theory/research_prerequisites.json',
    'theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md','theory/nodal/SINE_FORMED_CLASS_CONTACT.md'])
names.update('docs/assets/sine_formed_classes/two-port-compatibility-v1'+s for s in ('.json','.manifest.json','.protocol.json','.source.zip'))
names.update(p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.suffix in ('.py','.md'))
reviews=[OUT/'independent-launch-review.md',OUT/'independent-audit-review.md',OUT/'root-integration-review.md']
assert all(p.is_file() for p in reviews)
rows=[receipt(ROOT/n) for n in sorted(names)]
assert sum(r['bytes'] for r in rows)<64*1024**2
with ZipFile(OUT/'producing-source.zip','x',ZIP_DEFLATED) as archive:
    for n in sorted(names):archive.write(ROOT/n,n)
verify_archive_members(OUT/'producing-source.zip',{r['path']:r['sha256'] for r in rows},max_bytes=64*1024**2)
protocol={'schema':PROTOCOL_SCHEMA,'policy':declared_policy(),'head':baseline['head'],
    'runtime':{'python_executable':str(Path(sys.executable).resolve()),'python_prefix':str(Path(sys.prefix).resolve()),
               'python_version':sys.version,'package_root':str((ROOT/'src/tnfr').resolve())},
    'sources':rows,'archive':receipt(OUT/'producing-source.zip'),
    'reviews':[receipt(p) for p in reviews],'controls':receipt(OUT/'controls-result.json'),
    'worker_entry':receipt(OUT/'worker.py')}
write_json_once(OUT/'producing-protocol.json',protocol)
write_json_once(OUT/'producing-freeze.json',{'head':baseline['head'],'before_reserved_attempt':True,
    'source_archive':protocol['archive'],'protocol':receipt(OUT/'producing-protocol.json'),
    'source_records':rows,'reviews':protocol['reviews'],'controls':protocol['controls'],
    'reserved_attempts':1,'new_reference_steps_before_this_freeze':0})
print({'source_files':len(rows),'expanded_bytes':sum(r['bytes'] for r in rows),'protocol_bytes':(OUT/'producing-protocol.json').stat().st_size})
