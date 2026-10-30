"""Required guard and documentation checks; never rerun scientific producers."""
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from tnfr.utils.io import json_loads
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PY = sys.executable
OWNERS = ["theory/RESEARCH_PROGRAM.md", "theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md"]


def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())


def run(name, args):
    start = time.monotonic()
    command = [PY, *args]
    with (OUT / (name+".log")).open("xb") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    row = {"name": name, "command": command, "exit_code": result.returncode,
           "wall_seconds": time.monotonic()-start, "log": receipt(OUT / (name+".log"))}
    print({"check": name, "exit_code": result.returncode}, flush=True)
    return row


tasks = [
    ("prerequisite-ordinary", ["scripts/check_research_prerequisites.py"]),
    ("prerequisite-physical", ["scripts/check_research_prerequisites.py", "--require-physical-mapping"]),
    ("documentation-integrity", ["scripts/check_documentation.py"]),
    ("source-references", ["scripts/verify_internal_references.py", "--ci", "--dirs", *OWNERS]),
]
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(run, name, args) for name, args in tasks]
    results = [f.result() for f in futures]
assert [r["exit_code"] for r in results] == [0, 1, 0, 0]
registry = json_loads((ROOT / "theory/research_prerequisites.json").read_bytes())
assert registry["physical_mapping_readiness_review"] is None
write_json_once(OUT / "prerequisite-validation.json", {
    "passed": True, "checks": results[:2],
    "registry": receipt(ROOT / "theory/research_prerequisites.json"),
    "physical_mapping_readiness_review": None, "physical_mapping_authorized": False,
})
checks = results[2:]
for name, args in [
    ("documentation-stage", ["scripts/prepare_docs.py"]),
    ("mkdocs-strict", ["-m", "mkdocs", "build", "--strict"]),
    ("rendered-links", [str(OUT / "check_rendered.py")]),
]:
    row = run(name, args)
    assert row["exit_code"] == 0, name
    checks.append(row)
write_json_once(OUT / "documentation-validation.json", {
    "passed": True, "checks": checks,
    "validated_sources": [receipt(ROOT / name) for name in OWNERS],
    "new_scientific_evaluations_by_documentation_checks": 0,
})
