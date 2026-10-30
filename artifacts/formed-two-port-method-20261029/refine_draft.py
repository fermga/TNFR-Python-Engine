"""Use the smaller primary target record; retain the initial reviewed design."""
from pathlib import Path
from zipfile import ZipFile
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
old = (OUT / 'chapter-proposed.md').read_bytes()
before = b'Read the retained 32/64 endpoint brackets from the existing `two-port-capture-v1.json`'
after = b'Read the retained 32/64 endpoint brackets from the primary `two-port-compatibility-v1.json`'
assert old.count(before) == 1
new = old.replace(before, after)
with ZipFile(OUT / 'before-edit.zip') as archive:
    prefix = archive.read('theory/nodal/SINE_TWO_PORT_CAPTURE.md')
assert new.startswith(prefix)
for name, data in [('chapter-proposed-v2.md', new), ('chapter-append-v2.md', new[len(prefix):])]:
    with (OUT / name).open('xb') as output:
        output.write(data)
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
write_json_once(OUT / 'refinement.json', {
    'source': receipt(OUT / 'chapter-proposed.md'), 'final': receipt(OUT / 'chapter-proposed-v2.md'),
    'suffix': receipt(OUT / 'chapter-append-v2.md'), 'script': receipt(Path(__file__)),
    'target_input': receipt(ROOT / 'docs/assets/sine_formed_classes/two-port-compatibility-v1.json'),
    'change': 'Select the primary compatibility report for later primitive root readmission; avoid loading the unrelated 41-MB prepared capture chain.',
    'mathematical_candidate_changed': False})
print(receipt(OUT / 'chapter-proposed-v2.md'))
