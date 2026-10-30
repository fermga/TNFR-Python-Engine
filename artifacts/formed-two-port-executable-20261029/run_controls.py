"""One reviewed finite control attempt; retain failures without overwriting."""
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import tnfr
from tnfr.research.artifact_io import file_receipt, write_json_once
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
receipt = lambda p: file_receipt(p, label=p.relative_to(ROOT).as_posix())
freeze = json_loads((OUT / 'controls-freeze.json').read_bytes())
assert receipt(ROOT / freeze['archive']['path']) == freeze['archive']
for name in ('controls-junit.xml', 'controls-result.json', 'controls.log', 'controls-temp'):
    assert not (OUT/name).exists(), f'Existing control output: {name}'
for row in freeze['records']:
    assert receipt(ROOT / row['path']) == row, row['path']
command = [sys.executable, '-E', '-m', 'pytest', '-o', 'addopts=', '-q',
           'tests/research/test_formed_two_port_reference.py',
           'tests/research/test_formed_two_port_run.py',
           'tests/research/test_bounded_worker.py',
           'artifacts/formed-two-port-executable-20261029/admission_controls.py',
           '--junitxml=artifacts/formed-two-port-executable-20261029/controls-junit.xml',
           '--basetemp=artifacts/formed-two-port-executable-20261029/controls-temp']
start = time.monotonic()
timeout = False
with (OUT / 'controls.log').open('xb') as log:
    try:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code, timeout = None, True
counts = None
junit_parse_error = None
if (OUT/'controls-junit.xml').exists():
    try:
        suites = ET.parse(OUT/'controls-junit.xml').getroot()
        counts = {key: sum(int(s.get(key, 0)) for s in suites.iter('testsuite'))
                  for key in ('tests', 'failures', 'errors', 'skipped')}
    except (OSError, ET.ParseError, ValueError) as exc:
        junit_parse_error = f'{type(exc).__name__}: {exc}'
passed = code == 0 and not timeout and counts is not None and counts['tests'] > 0 and all(counts[k] == 0 for k in ('failures', 'errors', 'skipped'))
row = {'passed': passed, 'counts': counts, 'command': command, 'exit_code': code,
       'junit_parse_error': junit_parse_error,
       'runtime': {'version': sys.version, 'prefix': sys.prefix, 'flags': str(sys.flags), 'package': str(Path(tnfr.__file__).resolve())},
       'timed_out': timeout, 'timeout_seconds': 300, 'wall_seconds': time.monotonic()-start,
       'freeze': receipt(OUT/'controls-freeze.json'), 'log': receipt(OUT/'controls.log'),
       'reserved_reference_steps': 0, 'actual_source_transits': 0, 'root_searches': 0,
       'historical_producer_replays': 0}
if (OUT/'controls-junit.xml').exists():
    row['junit'] = receipt(OUT/'controls-junit.xml')
write_json_once(OUT/'controls-result.json', row)
print(row)
raise SystemExit(0 if row['passed'] else 1)
