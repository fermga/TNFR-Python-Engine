"""Preserve the admitted components before final launch integration."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess
import sys
import tnfr
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once, verify_archive_members

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'artifacts/formed-two-port-executable-20261029'
OWNERS=['src/tnfr/research/formed_two_port_run.py','tests/research/test_formed_two_port_run.py',
        'theory/nodal/SINE_TWO_PORT_CAPTURE.md','theory/RESEARCH_PROGRAM.md']
ADDED=['src/tnfr/research/formed_two_port_launch.py','tests/research/test_formed_two_port_launch.py']
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf-8')
prior=json_loads((PRIOR/'final-state.json').read_bytes())
assert prior['passed'] and git('rev-parse','HEAD').strip()==prior['head']
assert Path(tnfr.__file__).resolve().is_relative_to(ROOT/'src/tnfr')
for row in prior['records']:
    assert receipt(ROOT/row['path'])==row,row['path']
names={n for n in git('ls-files','-z','--cached','--others','--exclude-standard').split('\0') if n and (ROOT/n).is_file()}
current={n:sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
assert current==prior['source_sha256']
assert all(not(ROOT/n).exists() for n in ADDED)
with ZipFile(OUT/'before-edit.zip','x',ZIP_DEFLATED) as archive:
    for n in OWNERS: archive.write(ROOT/n,n)
verify_archive_members(OUT/'before-edit.zip',{n:current[n] for n in OWNERS})
write_json_once(OUT/'baseline.json',{
    'head':prior['head'],'source_sha256':current,
    'dirty_paths_before':git('status','--porcelain=v1','--untracked-files=all'),
    'prior_final_state':receipt(PRIOR/'final-state.json'),'prior_source_manifest':receipt(PRIOR/'source-manifest.json'),
    'prior_records_verified':len(prior['records']),'books':prior['books_preserved_in_place'],
    'selected_books':prior['selected_book_sources'],'archive':receipt(OUT/'before-edit.zip'),
    'archived_paths':OWNERS,'planned_added_paths':ADDED,'python':sys.executable,'version':sys.version,
    'prefix':sys.prefix,'package':str(Path(tnfr.__file__).resolve()),
    'scope':'Finalize/review/freeze launcher and protocol, then exactly one fixed reference attempt; no pilots/retries/mapping.'})
print({'source_files':len(current),'prior_records':len(prior['records']),'dirty_paths':len(git('status','--porcelain=v1','--untracked-files=all').splitlines())})
