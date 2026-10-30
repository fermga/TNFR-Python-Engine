"""Run only the prospectively frozen synthetic integration controls once."""
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
for name in ('controls.log','controls-junit.xml','controls-result.json','controls-temp'):
    assert not(OUT/name).exists(),name
freeze=json_loads((OUT/'controls-freeze.json').read_bytes())
for row in [*freeze['records'],freeze['archive']]:
    assert receipt(ROOT/row['path'])==row,row['path']
selected=['tests/research/test_formed_two_port_run.py','tests/research/test_formed_two_port_launch.py',
    'artifacts/formed-two-port-executable-20261029/admission_controls.py',
    'artifacts/formed-two-port-attempt-20261029/test_record_attempt.py',
    'artifacts/formed-two-port-attempt-20261029/test_audit_attempt.py']
command=[sys.executable,'-E','-m','pytest','-o','addopts=','-q',*selected,
    '--junitxml=artifacts/formed-two-port-attempt-20261029/controls-junit.xml',
    '--basetemp=artifacts/formed-two-port-attempt-20261029/controls-temp']
started=time.monotonic()
timeout=False
with (OUT/'controls.log').open('xb') as log:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        code=result.returncode
    except subprocess.TimeoutExpired:
        code,timeout=None,True
counts=None
parse_error=None
if (OUT/'controls-junit.xml').exists():
    try:
        suites=ET.parse(OUT/'controls-junit.xml').getroot()
        counts={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    except (OSError,ET.ParseError,ValueError) as exc:parse_error=f'{type(exc).__name__}: {exc}'
passed=code==0 and not timeout and counts is not None and counts['tests']>0 and all(counts[k]==0 for k in ('failures','errors','skipped'))
record={'passed':passed,'counts':counts,'command':command,'exit_code':code,'timed_out':timeout,
    'timeout_seconds':180,'wall_seconds':time.monotonic()-started,'junit_parse_error':parse_error,
    'runtime':{'python':sys.executable,'version':sys.version,'prefix':sys.prefix,'flags':str(sys.flags)},
    'freeze':receipt(OUT/'controls-freeze.json'),'log':receipt(OUT/'controls.log'),
    'reserved_reference_steps':0,'actual_source_transits':0,'target_sign_admissions':0,
    'root_searches':0,'historical_producer_replays':0,'real_workers_launched':0}
if (OUT/'controls-junit.xml').exists():record['junit']=receipt(OUT/'controls-junit.xml')
write_json_once(OUT/'controls-result.json',record)
print({'passed':passed,'counts':counts,'exit_code':code,'wall_seconds':record['wall_seconds']})
raise SystemExit(0 if passed else 1)
