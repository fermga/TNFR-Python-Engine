"""Frozen final caller wrapper: one bounded return, then the observed exit."""
from pathlib import Path
import sys
from tnfr.research.formed_two_port_launch import launch_formed_two_port, caller_return_bytes

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
try:
    result = launch_formed_two_port(ROOT, OUT/'producing-protocol.json', OUT/'worker.py', OUT/'attempt')
except Exception as exc:
    result = {'record_stage':'launcher_exception', 'execution_passed':False,
              'exception_type':type(exc).__name__, 'reason':str(exc)[:2048]}
sys.stdout.buffer.write(caller_return_bytes(result))
sys.stdout.buffer.flush()
raise SystemExit(0 if result.get('execution_passed') is True else 2)
