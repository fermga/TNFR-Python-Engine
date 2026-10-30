"""One separately reviewed post-hoc stored-data diagnostic, never a rerun."""
from pathlib import Path
import subprocess
import sys
import time
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
for name in ('stored-tube-diagnostic.json','diagnostic.log','diagnostic-execution.json','diagnostic-freeze.json'):
    assert not(OUT/name).exists(),name
script=receipt(OUT/'diagnose_stored_tubes.py')
assert script['sha256']=='f1834bc8cdce75f8bf9d261f50a06b6a1292ad8359280c68c1b94059c2b27b51'
plan=receipt(OUT/'diagnostic_plan.md')
assert plan['sha256']=='4f0a76aa6c09665c0a6d14fd5e49439ecf737b994a2a37d6ce229bdee4ef6a3f'
review=receipt(OUT/'diagnostic-review.md')
write_json_once(OUT/'diagnostic-freeze.json',{'before_diagnostic':True,'post_hoc':True,
    'script':script,'plan':plan,'review':review,'wrapper':receipt(Path(__file__).resolve()),
    'new_reference_steps_authorized':False,'timeout_seconds':30})
started=time.monotonic()
timed_out=False
with (OUT/'diagnostic.log').open('xb') as log:
    try:
        result=subprocess.run([sys.executable,'-E',str(OUT/'diagnose_stored_tubes.py')],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,timeout=30)
        code=result.returncode
    except subprocess.TimeoutExpired:
        timed_out,code=True,None
record={'passed':code==0 and not timed_out,'post_hoc':True,'exit_code':code,'timed_out':timed_out,
    'timeout_seconds':30,'wall_seconds':time.monotonic()-started,'log':receipt(OUT/'diagnostic.log'),
    'freeze':receipt(OUT/'diagnostic-freeze.json'),'scientific_producer_calls':0,'new_reference_steps':0}
if (OUT/'stored-tube-diagnostic.json').exists():
    record['output']=receipt(OUT/'stored-tube-diagnostic.json')
write_json_once(OUT/'diagnostic-execution.json',record)
print({'passed':record['passed'],'exit_code':code,'post_hoc':True,'new_reference_steps':0})
