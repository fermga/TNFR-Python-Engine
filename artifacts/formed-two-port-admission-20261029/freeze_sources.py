"""Bind reviewed mathematical premises and exact controls before execution."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once, verify_archive_members
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
baseline=json_loads((OUT/'baseline.json').read_bytes())
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
for name,digest in baseline['source_sha256'].items():
    assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
names=['AGENTS.md','theory/research_prerequisites.json','theory/RESEARCH_PROGRAM.md',
    'theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md','theory/nodal/SINE_TWO_PORT_CAPTURE.md',
    'theory/nodal/SINE_FORMED_CLASS_CONTACT.md','theory/nodal/SINE_FORMED_CLASS_MAINTENANCE.md',
    'theory/nodal/SINE_TWO_PORT_DIPOLE.md','src/tnfr/physics/relational_sine_two_port_dipole.py',
    'src/tnfr/physics/_sine_lyapunov.py','src/tnfr/physics/_sine_formed_contact.py',
    'src/tnfr/research/artifact_io.py','src/tnfr/utils/io.py']
names += [(OUT/n).relative_to(ROOT).as_posix() for n in (
    'scope-and-protocol.md','chapter-append-v2.md','chapter-proposed-v2.md','program-proposed.md',
    'check_admission.py','run_exact_controls.py','algebra-review.md','integration-review.md','response-review.md','response-review-final.md',
    'source-event-analysis.md','identity-analysis.md','response-analysis.md')]
records=[receipt(ROOT/n) for n in names]
with ZipFile(OUT/'reviewed-source.zip','x',ZIP_DEFLATED) as z:
    for n in names:z.write(ROOT/n,n)
verify_archive_members(OUT/'reviewed-source.zip',{r['path']:r['sha256'] for r in records})
write_json_once(OUT/'source-freeze.json',{'records':records,'archive':receipt(OUT/'reviewed-source.zip'),
    'head':baseline['head'],'script':receipt(Path(__file__)),
    'before_exact_controls':True,'actual_source_transit_authorized':False})
print({'frozen_files':len(records),'actual_source_transit_authorized':False})
