"""Freeze reviewed method premises and finite controls before their one run."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once, verify_archive_members

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
baseline = json_loads((OUT / 'baseline.json').read_bytes())
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
for name, digest in baseline['source_sha256'].items():
    assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
names = ['AGENTS.md', 'theory/research_prerequisites.json', 'theory/RESEARCH_PROGRAM.md',
    'theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md', 'theory/nodal/SINE_TWO_PORT_CAPTURE.md',
    'theory/nodal/SINE_TWO_PORT_TRANSIT.md', 'theory/nodal/SINE_FORMED_CLASS_CONTACT.md',
    'theory/nodal/SINE_FORM_PHASE_REDUCTION.md',
    'src/tnfr/physics/relational_sine_two_port_compatibility.py',
    'src/tnfr/physics/relational_sine_two_port_capture.py',
    'src/tnfr/research/sine_two_port_handoff.py',
    'src/tnfr/mathematics/_validated_metric.py', 'src/tnfr/mathematics/_validated_taylor.py',
    'src/tnfr/mathematics/_comparison_flow.py', 'src/tnfr/mathematics/_interval_taylor.py',
    'src/tnfr/mathematics/_rational_interval.py',
    'src/tnfr/research/artifact_io.py', 'src/tnfr/utils/io.py',
    'docs/assets/sine_formed_classes/two-port-compatibility-v1.json',
    'docs/assets/sine_formed_classes/two-port-compatibility-v1.protocol.json',
    'docs/assets/sine_formed_classes/two-port-compatibility-v1.manifest.json',
    'docs/assets/sine_formed_classes/two-port-compatibility-v1.source.zip']
names += [(OUT / n).relative_to(ROOT).as_posix() for n in (
    'chapter-proposed-v3.md', 'chapter-append-v3.md', 'program-proposed.md',
    'controls-protocol.md', 'check_method.py', 'run_exact_controls.py',
    'slow-reference-analysis.md', 'method-kernel-analysis.md', 'domain-analysis.md',
    'integration-review.md', 'domain-review.md', 'controls-review.md')]
records = [receipt(ROOT / n) for n in names]
with ZipFile(OUT / 'reviewed-source.zip', 'x', ZIP_DEFLATED) as archive:
    for n in names:
        archive.write(ROOT / n, n)
verify_archive_members(OUT / 'reviewed-source.zip', {r['path']: r['sha256'] for r in records})
write_json_once(OUT / 'source-freeze.json', {
    'records': records, 'archive': receipt(OUT / 'reviewed-source.zip'),
    'head': baseline['head'], 'script': receipt(Path(__file__)),
    'before_exact_controls': True, 'actual_source_transit_authorized': False,
    'boundary': 'Freeze for finite rational method controls only; future reference adapter/supervisor implementation and source freeze remain open.'})
print({'frozen_files': len(records), 'actual_source_transit_authorized': False})
