import sys
import os
import runpy

# Ensure local 'tnfr' package from src is used
repo_src = r"c:/TNFR-Python-Engine/src"
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

# Set a larger N for timing and export (adjust via env override)
DEFAULT_N = '5000'
os.environ['TNFR_ARITH_MAX_N'] = os.environ.get('TNFR_ARITH_MAX_N', DEFAULT_N)

if __name__ == "__main__":
    runpy.run_path('benchmarks/arith_fields_export.py', run_name='__main__')
