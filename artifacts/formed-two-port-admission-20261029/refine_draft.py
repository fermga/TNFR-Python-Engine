"""Clarify fixed lift transport without changing the admitted mathematics."""
from pathlib import Path
from zipfile import ZipFile
from tnfr.research.artifact_io import file_receipt, write_json_once
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
old=(OUT/'chapter-proposed.md').read_bytes()
changes=[(
'''`-4*pi`/`-2*pi`, respectively. Adding those fixed integer turns restores
the old increasing-ramp chart; move every edge offset, residual, observer
''',
'''`-4*pi`/`-2*pi`, respectively. Add `+4*pi` to those donor lifts and
`+2*pi` to those receiver lifts to restore the old increasing-ramp chart;
move every edge offset, residual, observer
'''),(
'''available before contact, afterward and in the disconnected control.
Let `q*` be the compatible target's internal gaps transported to these labels.
''',
'''available before contact, afterward and in the disconnected control.
They are continuous lifted increments, without repeated principal-value
wrapping. In the admitted acute collar they agree with the corresponding
circular edge representatives; leaving that chart does not reset a lift.
Let `q*` be the compatible target's internal gaps transported to these labels.
''')]
new=old
for a,b in changes:
    a,b=a.replace('\n','\r\n').encode(),b.replace('\n','\r\n').encode()
    assert new.count(a)==1
    new=new.replace(a,b)
with (OUT/'chapter-proposed-v2.md').open('xb') as f:f.write(new)
with ZipFile(OUT/'before-edit.zip') as z:prefix=z.read('theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md')
assert new.startswith(prefix)
with (OUT/'chapter-append-v2.md').open('xb') as f:f.write(new[len(prefix):])
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
write_json_once(OUT/'refinement.json',{'changes':changes,'previous':receipt(OUT/'chapter-proposed.md'),
    'updated':receipt(OUT/'chapter-proposed-v2.md'),'append':receipt(OUT/'chapter-append-v2.md'),
    'program_unchanged':receipt(OUT/'program-proposed.md'),'script':receipt(Path(__file__))})
print(receipt(OUT/'chapter-proposed-v2.md'))
