"""Apply only the two exact reviewed method drafts after finite controls."""
from hashlib import sha256
from pathlib import Path
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
read = lambda n: json_loads((OUT / n).read_bytes())
for name, digest in read('baseline.json')['source_sha256'].items():
    if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise ValueError(name)
if not read('method-controls.json')['passed'] or not read('exact-execution.json')['passed']:
    raise ValueError('Finite method controls have not passed')
for row in read('source-freeze.json')['records']:
    if receipt(ROOT / row['path']) != row:
        raise ValueError(row['path'])
pairs = [('chapter-proposed-v3.md', 'theory/nodal/SINE_TWO_PORT_CAPTURE.md'),
         ('program-proposed.md', 'theory/RESEARCH_PROGRAM.md')]
hashes = ['ab36ac0738d1dfa707f5adf45a52ee2c6245dd2d3efe6b3e9bb4cfa2377c2338',
          'abee58d40fe276d4794ac8adab4d77880e79d976485d9f3a0e1976ee58f89d64']
for (proposal, maintained), digest in zip(pairs, hashes):
    if receipt(OUT / proposal)['sha256'] != digest:
        raise ValueError(proposal)
# Root must read the actual substantive reviews, not rely on hash presence.
for name in ('integration-review.md', 'domain-review.md', 'controls-review.md'):
    text = (OUT / name).read_text(encoding='utf-8')
    if any(h not in text for h in hashes):
        raise ValueError(name)
for proposal, maintained in pairs:
    (ROOT / maintained).write_bytes((OUT / proposal).read_bytes())
write_json_once(OUT / 'applied-integration.json', {
    'associations': [receipt(OUT / n) for n in (
        'baseline.json', 'before-edit.zip', 'drafts.json', 'refinement.json', 'proof-refinement.json',
        'build_drafts.py', 'refine_draft.py', 'refine_proof.py', 'source-freeze.json', 'reviewed-source.zip',
        'controls-protocol.md', 'integration-review.md', 'domain-review.md', 'controls-review.md',
        'check_method.py', 'run_exact_controls.py', 'method-controls.json', 'exact-execution.json',
        'apply_integration.py')],
    'files': [{'proposal': receipt(OUT / p), 'maintained': receipt(ROOT / m)} for p, m in pairs],
    'new_analytic_method': True, 'actual_source_transits': 0, 'reference_steps': 0,
    'root_solves': 0, 'historical_producer_replays': 0,
    'next_question': 'Implement and independently review the source-specific reference adapter and bounded supervisor; freeze actual producing source before one reserved transit.'})
print({'changed': [m for p, m in pairs], 'reference_steps': 0})
