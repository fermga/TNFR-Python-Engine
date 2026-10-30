"""One separate post-run documentary audit; never restart the reference."""
from pathlib import Path
import subprocess
import sys
import time
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
for name in ('audit.json','audit.log','audit-execution.json'):
    assert not(OUT/name).exists(),name
command=[sys.executable,'-E',str(OUT/'audit_attempt.py'),
    '--attempt',str(OUT/'attempt'),'--protocol',str(OUT/'producing-protocol.json'),
    '--caller-result',str(OUT/'caller-result.json'),'--output',str(OUT/'audit.json')]
started=time.monotonic()
timed_out=False
with (OUT/'audit.log').open('xb') as log:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        code=result.returncode
    except subprocess.TimeoutExpired:
        timed_out,code=True,None
record={'completed':not timed_out,'exit_code':code,'timed_out':timed_out,
    'documentary_only':True,'timeout_seconds':120,'wall_seconds':time.monotonic()-started,
    'command':command,'script':receipt(OUT/'audit_attempt.py'),'log':receipt(OUT/'audit.log'),
    'protocol':receipt(OUT/'producing-protocol.json'),'caller_result':receipt(OUT/'caller-result.json'),
    'new_reference_steps':0,'new_target_sign_admissions':0}
if (OUT/'audit.json').exists():record['audit']=receipt(OUT/'audit.json')
write_json_once(OUT/'audit-execution.json',record)
print({'exit_code':code,'timed_out':timed_out,'documentary_only':True})
