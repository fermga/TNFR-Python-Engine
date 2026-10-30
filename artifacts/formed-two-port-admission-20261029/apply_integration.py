"""Apply reviewed admission and update the sole queue after exact checks."""
from hashlib import sha256
from pathlib import Path
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
read=lambda n:json_loads((OUT/n).read_bytes())
baseline=read('baseline.json')
for name,digest in baseline['source_sha256'].items():
    if sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError(name)
if not read('algebra-checks.json')['passed'] or not read('exact-execution.json')['passed']:
    raise ValueError('Exact controls have not passed')
for row in read('source-freeze.json')['records']:
    if receipt(ROOT/row['path'])!=row:raise ValueError(row['path'])
pairs=[('chapter-proposed-v2.md','theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md'),
       ('program-proposed.md','theory/RESEARCH_PROGRAM.md')]
hashes=['b6a17ba53681a17111c2cb960ac01224596907ffa77a733229c115c322cc4607',
        'ee00bcc4147cfb7ae52f080ba495542924e74dc8347431031357a256bf7bc960']
for (p,m),digest in zip(pairs,hashes):
    if receipt(OUT/p)['sha256']!=digest:raise ValueError(p)
# Root reads substantive reviews before running this; these checks bind bytes.
for name in ('integration-review.md','response-review-final.md'):
    text=(OUT/name).read_text(encoding='utf-8')
    if any(h not in text for h in hashes):raise ValueError(name)
for p,m in pairs:(ROOT/m).write_bytes((OUT/p).read_bytes())
write_json_once(OUT/'applied-integration.json',{
    'associations':[receipt(OUT/n) for n in ('baseline.json','before-edit.zip',
        'drafts.json','refinement.json','build_drafts.py','refine_draft.py',
        'source-freeze.json','reviewed-source.zip','scope-and-protocol.md',
        'algebra-review.md','integration-review.md','response-review.md','response-review-final.md',
        'check_admission.py','run_exact_controls.py','algebra-checks.json','exact-execution.json','apply_integration.py')],
    'files':[{'proposal':receipt(OUT/p),'maintained':receipt(ROOT/m)} for p,m in pairs],
    'actual_source_transits':0,'historical_producer_replays':0,
    'new_analytic_admission':True,'next_question':'source-aware entry method and independently reviewed prospective protocol'})
print({'changed':[m for p,m in pairs],'actual_source_transits':0})
