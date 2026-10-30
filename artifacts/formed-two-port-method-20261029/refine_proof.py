"""Preserve v2 and make the reviewed endpoint implication explicit."""
from pathlib import Path
from zipfile import ZipFile
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
data = (OUT / 'chapter-proposed-v2.md').read_bytes()
changes = [
    (b'while `||theta(T+1)-theta_*^m||_D<Q+rho<1/1024`.',
     b'while `||theta(T+1)-theta_*^m||_D<=Q+rho<1/1024`.'),
    (b'\\mathcal R^2<65/67108864<1/648000=B,',
     b'H-H_*\\le\\mathcal R^2<65/67108864<1/648000=B,')]
for old, new in changes:
    assert data.count(old) == 1
    data = data.replace(old, new)
with ZipFile(OUT / 'before-edit.zip') as archive:
    prefix = archive.read('theory/nodal/SINE_TWO_PORT_CAPTURE.md')
assert data.startswith(prefix)
for name, contents in [('chapter-proposed-v3.md', data), ('chapter-append-v3.md', data[len(prefix):])]:
    with (OUT / name).open('xb') as output:
        output.write(contents)
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
write_json_once(OUT / 'proof-refinement.json', {
    'source': receipt(OUT / 'chapter-proposed-v2.md'), 'final': receipt(OUT / 'chapter-proposed-v3.md'),
    'suffix': receipt(OUT / 'chapter-append-v3.md'), 'script': receipt(Path(__file__)),
    'change': 'Use the consumed non-strict phase inequality and display the endpoint excess-storage inequality explicitly in M9.',
    'candidate_or_budget_changed': False})
print(receipt(OUT / 'chapter-proposed-v3.md'))
