"""Preserve the original scope review and record the clarified final draft."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())

original_path = OUT / 'scope-review.json'
payload = json_loads(original_path.read_bytes())
clarification_path = OUT / 'clarification-record.json'
clarification = json_loads(clarification_path.read_bytes())
archive_path = ROOT / clarification['archive']['path']
assert receipt(archive_path) == clarification['archive']
assert any(item == receipt(original_path) for item in clarification['superseded_reviews'])
with ZipFile(archive_path) as archive:
    for item in payload['reviewed_files']:
        archived = archive.read(Path(item['path']).name)
        assert sha256(archived).hexdigest() == item['sha256']
        assert len(archived) == item['bytes']
    for name in ('admission-section.md', 'program-introduction.md'):
        assert archive.read(name) == (OUT / name).read_bytes()
    old_program = archive.read('program-decision.md').decode('utf-8')
new_program = (OUT / 'program-decision.md').read_text(encoding='utf-8')
old_program = old_program.replace('\r\n', '\n')
expected = old_program.replace(
    'URL, family and access-based selection reason before detailed assessment.\n',
    'URL, family and access-based selection reason before detailed assessment.\n'
    'Record access date and version/DOI/revision/commit when available, and the\n'
    'section or page supporting each extracted fact. Unknown metadata remain\n'
    'unknown rather than being inferred from another source.\n',
).replace(
    'Stop when the selected cases are classified or either source/query limit\n'
    'is reached. Close the audit even if no candidate survives. Retain the source\n',
    'Stop new searches and document selections when either source/query limit\n'
    'is reached; finish classifying only the material already admitted, without\n'
    'extra retrievals or queries. Close the audit when those cases are classified,\n'
    'even if no candidate survives. Retain the source\n',
)
assert expected == new_program
payload['reviewed_files'] = [
    receipt(OUT / name)
    for name in ('admission-section.md', 'program-introduction.md', 'program-decision.md')
]
payload['supersedes'] = receipt(original_path)
payload['clarification_record'] = receipt(clarification_path)
payload['prior_draft_archive'] = receipt(archive_path)
payload['clarification_assessment'] = (
    'PASS after re-reading the final program draft. The only changes are explicit source provenance '
    'requirements and clarification that reaching either budget stops new acquisition while classification '
    'is completed only from already admitted material, without extra retrievals or queries. '
    'The six-document/six-query ceiling, three fixed apparatus families, one apparatus per family, '
    'allowed conclusions, physical activity marker and exclusions are unchanged. Missing version or '
    'location metadata remain unavailable rather than being invented. This refinement does not expand '
    'the source budget or authorize an empirical response.'
)
payload['assessments']['bounded_selection'] += (
    ' The final draft additionally requires access date, available version/DOI/revision/commit, and '
    'section/page support for each extracted fact, retaining unknown metadata explicitly.'
)
payload['assessments']['closure_and_followup'] = (
    'New searches and document selections stop when either ceiling is reached. Classification may finish '
    'only for already admitted material, without extra retrievals or queries. The audit closes even if no '
    'candidate survives, retains one source ledger and one physical-assessment result, and recommends at '
    'most one separate bridge review or concrete dependency. PROGRAM remains the sole queue. No follow-up '
    'mapping, fitting, numerical response, expanded budget or changed consumer is implicitly authorized; '
    'each needs a new exact-task admission and applicable prospective evaluation controls.'
)
payload['writer'] = receipt(Path(__file__).resolve())
write_json_once(OUT / 'scope-review-final.json', payload)
print(receipt(OUT / 'scope-review-final.json'))
