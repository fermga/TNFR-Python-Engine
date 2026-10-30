"""Assemble the source-aware method without changing preserved proof bytes."""
from pathlib import Path
from zipfile import ZipFile
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CHAPTER = 'theory/nodal/SINE_TWO_PORT_CAPTURE.md'
PROGRAM = 'theory/RESEARCH_PROGRAM.md'
with ZipFile(OUT / 'before-edit.zip') as archive:
    chapter, program = archive.read(CHAPTER), archive.read(PROGRAM)
encode = lambda text, old: text.replace('\n', '\r\n' if b'\r\n' in old else '\n').encode()
chapter += encode((OUT / 'chapter-append.md').read_text(encoding='utf-8'), chapter)
edits = []


def replace(old, new):
    global program
    a, b = encode(old, program), encode(new, program)
    assert program.count(a) == 1, old[:100]
    program = program.replace(a, b)
    edits.append({'old': old, 'new': new})


replace('''with an initial-absence margin and supplied work bound. What remains is a
source-aware entry method under the same complete law. No new transit,
trajectory or source-to-target result has been evaluated.
''', '''with an initial-absence margin and supplied work bound. The
[source-aware reference method](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-entry-method)
now closes the analytical comparison and fixes a prospective numerical policy.
Its source-specific adapter, bounded supervisor and source freeze remain the
next obligation. No new transit or source-to-target result has been evaluated.
''')
replace('Source/event/identity admission is closed at the [formed-source contract](nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-admission). Select one source-aware entry method and freeze its protocol before transit evaluation.',
        'The [formed-source contract](nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-admission) and [source-aware reference method](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-entry-method) are established within their conditional scopes. Implement and review the selected bounded adapter, then freeze producing source before one reserved transit.')
start = b'### Sole active question and first obligation'
end = b'### Source context and available methods'
assert program.count(start) == program.count(end) == 1
program = program[:program.index(start)] + encode((OUT / 'program-current.md').read_text(encoding='utf-8'), program) + program[program.index(end):]
replace("Carry the useful intersections into the selected entry-method declaration:",
        "Carry the useful intersections into the selected method's executable admission:")
assert len(program) <= 96 * 1024 and len(chapter.splitlines()) <= 4000
for name, data in [('chapter-proposed.md', chapter), ('program-proposed.md', program)]:
    with (OUT / name).open('xb') as output:
        output.write(data)
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
write_json_once(OUT / 'drafts.json', {'replacements': edits, 'sources': [receipt(OUT / n) for n in
    ('before-edit.zip', 'chapter-append.md', 'program-current.md', 'build_drafts.py', 'chapter-proposed.md', 'program-proposed.md')]})
print({'chapter_lines': len(chapter.splitlines()), 'program_bytes': len(program),
       'drafts': [receipt(OUT / n) for n in ('chapter-proposed.md', 'program-proposed.md')]})
