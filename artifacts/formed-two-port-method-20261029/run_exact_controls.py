"""Run the frozen finite algebra once, with active assertions and a hard cap."""
from pathlib import Path
import subprocess
import sys
import time
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
receipt=lambda p:file_receipt(p,label=p.relative_to(ROOT).as_posix())
freeze=json_loads((OUT/'source-freeze.json').read_bytes())
for row in freeze['records']:
    actual=receipt(ROOT/row['path'])
    if actual!=row:raise ValueError(f"Frozen input changed: {row['path']}")
# -E ignores PYTHONOPTIMIZE (and other Python environment flags); no -O or
# optimized interpreter flag is passed. The checked child executes assertions.
command=[sys.executable,'-E',str(OUT/'check_method.py')]
start=time.monotonic();timed_out=False
with (OUT/'exact-controls.log').open('xb') as log:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                              timeout=60,check=False)
        code=result.returncode
    except subprocess.TimeoutExpired:
        timed_out=True;code=None
write_json_once(OUT/'exact-execution.json',{'passed':code==0 and not timed_out,
    'command':command,'timeout_seconds':60,'timed_out':timed_out,'exit_code':code,
    'wall_seconds':time.monotonic()-start,'assertions':'enabled: -E, no -O',
    'source_freeze':receipt(OUT/'source-freeze.json'),'log':receipt(OUT/'exact-controls.log'),
    'wrapper':receipt(Path(__file__))})
if code!=0 or timed_out:raise RuntimeError('Exact control attempt failed; preserve and review before any correction.')
print((OUT/'exact-controls.log').read_text(encoding='utf-8').strip())
