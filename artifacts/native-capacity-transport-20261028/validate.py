"""Freeze one reviewed native capacity-transport source, then supervise it once.

The existing stdlib Windows supervisor is reused as a process helper only.
Its 120-second watchdog and sampled aggregate RSS allowance are enforced by
terminating the owned job; the RSS check is not a hard OS allocation limit.
No historical producer, runtime integration or trajectory is executed.
"""

import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv313/Scripts/python.exe"
HELPER = ROOT / "artifacts/parent-child-cut-direct-response-20261023"
TEST = "tests/physics/test_native_capacity_transport.py"
PROOF = "theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md"
ALLOWED = {TEST, PROOF, "theory/RESEARCH_PROGRAM.md"}
NAMES = (
    'test_measure_family_and_cycle_equivalence',
    'test_weighted_balance_variance_and_positive_generator',
    'test_native_storage_cancellation_on_joined_support',
    'test_port_contrast_discriminates_means_and_cut_balances',
    'test_matched_native_acceleration_and_blind_cases',
    'test_link_change_separates_measure_and_storage_jumps',
)

BUDGET = {
    "watchdog_wall_budget_seconds": 120,
    "maximum_rss_bytes": 2 * 1024**3,
    "maximum_response_bytes": 32 * 1024**2,
    "maximum_log_bytes": 1024**2,
    "diagnostic_reserve_bytes": 1024**2,
    "handshake_timeout_seconds": 30,
    "poll_seconds": 0.1,
}
RESOURCE_SCOPE = (
    "Windows kill-on-close job; aggregate RSS sampled with a nominal 100 ms polling interval, "
    "with actual-worker OS peak additionally checked; no hard allocation limit. "
    "The 120-second watchdog includes the interpreter handshake and worker setup; "
    "owned-process cleanup has its separate bounded allowance."
)
for directory in (ROOT, ROOT / "src"):
    if str(directory) in sys.path:
        sys.path.remove(str(directory))
    sys.path.insert(0, str(directory))

from tnfr.research.artifact_io import (
    file_receipt,
    verify_archive_members,
    write_json_once,
)
from tnfr.utils.io import json_loads


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(path):
    path = Path(path)
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())


def read(name):
    return json_loads((OUT / name).read_bytes())


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    ).stdout


def inventory():
    return sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "src/tnfr").rglob("*.py")
    )


def verify_baseline():
    before = read("baseline.json")
    require(
        set(before["allowed_changed_paths"]) == ALLOWED, "baseline allowlist differs"
    )
    require(git("rev-parse", "HEAD").strip() == before["head"], "HEAD changed")
    for name, digest in before["source_sha256"].items():
        if name not in ALLOWED:
            require(
                receipt(ROOT / name)["sha256"] == digest,
                "baseline source changed: " + name,
            )
    initial = {line[3:]: line[:2] for line in before["dirty_paths_before"].splitlines()}
    current = {
        line[3:]: line[:2]
        for line in git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
    }
    changed = {
        name
        for name in set(initial) | set(current)
        if initial.get(name) != current.get(name)
    }
    require(
        changed <= ALLOWED,
        "unexpected working-tree change: " + str(sorted(changed - ALLOWED)),
    )
    return before


def verify_declaration():
    scope, prospective = read("scope.json"), read("prospective-run.json")
    require(
        prospective["scope"] == receipt(OUT / "scope.json"), "prospective scope changed"
    )
    require(
        prospective["independent_review"]
        == receipt(OUT / "pre-evaluation-review.json"),
        "prospective independent review changed",
    )
    require(
        scope["evaluation_budget"]
        == {
            "pytest_attempts": 1,
            "wall_seconds": 120,
            "memory_bytes": 2 * 1024**3,
            "trajectory_runs": 0,
            "parameter_scans": 0,
        },
        "scientific scope budget changed",
    )
    require(
        prospective["expected_test_names"] == list(NAMES), "declared test names differ"
    )
    require(
        prospective["expected_test_count"] == 6 and prospective["pytest_attempts"] == 1,
        "attempt or count differs",
    )
    require(prospective["budget"] == BUDGET, "supervisor budget differs")
    require(prospective["resource_scope"] == RESOURCE_SCOPE, "resource scope differs")
    require(
        prospective["allowed_post_evaluation_status_edits"]
        == ["theory/RESEARCH_PROGRAM.md"],
        "post-evaluation allowance differs",
    )
    tree = ast.parse((ROOT / TEST).read_text(encoding="utf-8"))
    tests = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")
    ]
    require(
        [item.name for item in tests] == list(NAMES), "actual test definitions differ"
    )
    require(not any(item.decorator_list for item in tests), "unexpected test decorator")
    return prospective


def environment():
    import tnfr

    require(
        Path(sys.executable).resolve() == PYTHON.resolve(), "wrong Python interpreter"
    )
    require(
        Path(tnfr.__file__).resolve() == (ROOT / "src/tnfr/__init__.py").resolve(),
        "TNFR is not the live checkout",
    )
    return {
        "python": str(Path(sys.executable).resolve()),
        "version": sys.version,
        "base_executable": str(Path(sys._base_executable).resolve()),
        "prefix": sys.prefix,
        "tnfr_source": str(Path(tnfr.__file__).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("pytest", "sympy", "numpy", "networkx")
        },
        "pytest_plugin_autoload": False,
        "pytest_addopts": "",
    }


def selected_paths():
    paths = set(inventory()) | ALLOWED
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("conftest.py")
    )
    paths.update(
        {
            "AGENTS.md",
            ".github/agents/my-agent.md",
            "README.md",
            "TESTING.md",
            "pyproject.toml",
            "theory/research_prerequisites.json",
            "theory/research_catalog.json",
            "tests/physics/test_native_capacity_foundation.py",
            "artifacts/native-model-admission-20261028/final-state.json",
            "theory/NODAL_PARAMETER_FOUNDATIONS.md",
            "theory/STRUCTURAL_OPERATORS.md",
            "theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md",
            "theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md",
            "theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md",
            "theory/CAPACITY_LOCALIZATION_BALANCE.md",
            "theory/TNFR_DIFFUSION_STABILITY_THEOREM.md",
            "theory/nodal/OPERATOR_PULSE_CLOSURE.md",
            "theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md",
            "theory/nodal/SINE_COLLECTIVE_PRESSURE_HANDOFF.md",
            "theory/nodal/SINE_PRESSURE_COMPLETION.md",
            "theory/NATIVE_OPERATOR_DYNAMICS.md",
            "theory/FORCED_SUPPORT_BALANCE.md",
            "scripts/check_research_prerequisites.py",
            "scripts/verify_internal_references.py",
        }
    )
    for name in (
        "conftest.py",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
        "tests/__init__.py",
        "tests/physics/__init__.py",
    ):
        if (ROOT / name).is_file():
            paths.add(name)
    paths.update(
        (HELPER / name).relative_to(ROOT).as_posix()
        for name in ("supervision.py", "bootstrap.py")
    )
    paths.update(
        (OUT / name).relative_to(ROOT).as_posix()
        for name in (
            "validate.py",
            "baseline.json",
            "before-edit.zip",
            "scope.json",
            "prospective-run.json",
            "pre-evaluation-review.json",
        )
    )
    return sorted(paths)


def pytest_arguments():
    return [
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-o",
        "junit_family=legacy",
        "--junitxml=" + str(OUT / "control-results.xml"),
        *[TEST + "::" + name for name in NAMES],
    ]


def freeze():
    before = verify_baseline()
    verify_declaration()
    review = read("pre-evaluation-review.json")
    require(
        review["schema"] == "tnfr.scientific-pre-evaluation-review.v1"
        and review["verdict"] == "PASS",
        "actual independent source review must pass",
    )
    require(
        review["reviewer"] == "/root/collective_controls",
        "independent reviewer differs",
    )
    require(
        {row["path"] for row in review["sources"]} == {TEST, PROOF}
        and len(review["sources"]) == 2,
        "reviewed proof/control set differs",
    )
    require(review["scope"] == receipt(OUT / "scope.json"), "review scope changed")
    require(
        review["scientific_executions_by_reviewer"] == 0
        and review["physical_mapping_authorized"] is False,
        "review boundary differs",
    )
    for row in review["sources"]:
        require(
            receipt(ROOT / row["path"]) == row,
            "reviewed source changed: " + row["path"],
        )
    actual_environment = environment()
    sources = [receipt(ROOT / name) for name in selected_paths()]
    with ZipFile(OUT / "evaluated-source.zip", "x", ZIP_DEFLATED) as archive:
        for row in sources:
            archive.write(ROOT / row["path"], row["path"])
    verify_archive_members(
        OUT / "evaluated-source.zip", {row["path"]: row["sha256"] for row in sources}
    )
    write_json_once(
        OUT / "validation-plan.json",
        {
            "frozen_utc": utc(),
            "head": before["head"],
            "source": sources,
            "source_archive": receipt(OUT / "evaluated-source.zip"),
            "review": receipt(OUT / "pre-evaluation-review.json"),
            "scope": receipt(OUT / "scope.json"),
            "prospective": receipt(OUT / "prospective-run.json"),
            "environment": actual_environment,
            "native_python_inventory": inventory(),
            "expected_test_names": list(NAMES),
            "expected_test_count": 6,
            "pytest_arguments": pytest_arguments(),
            "budget": BUDGET,
            "resource_scope": RESOURCE_SCOPE,
            "pytest_attempts": 1,
            "trajectory_runs": 0,
            "parameter_scans": 0,
            "prior_producer_replays": 0,
        },
    )
    print({"status": "reviewed_source_frozen", "sources": len(sources), "controls": 6})


def verify_frozen(plan):
    require(plan["expected_test_names"] == list(NAMES), "frozen names differ")
    require(
        plan["pytest_arguments"] == pytest_arguments(), "frozen pytest arguments differ"
    )
    require(plan["budget"] == BUDGET, "frozen budget differs")
    require(
        plan["native_python_inventory"] == inventory(),
        "native source inventory changed",
    )
    require(plan["environment"] == environment(), "execution environment changed")
    require(git("rev-parse", "HEAD").strip() == plan["head"], "frozen HEAD changed")
    for row in plan["source"] + [
        plan["review"],
        plan["scope"],
        plan["prospective"],
        plan["source_archive"],
    ]:
        require(
            receipt(ROOT / row["path"]) == row, "frozen source changed: " + row["path"]
        )
    verify_archive_members(
        ROOT / plan["source_archive"]["path"],
        {row["path"]: row["sha256"] for row in plan["source"]},
    )


def supervisor():
    spec = importlib.util.spec_from_file_location(
        "native_capacity_supervision", HELPER / "supervision.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker():
    marker = globals().get("__tnfr_supervision__")
    supervisor().validate_bootstrap_admission(
        marker, expected_prefix=ROOT / ".venv313", expected_executable=PYTHON
    )
    plan = read("validation-plan.json")
    verify_frozen(plan)
    start = read("control-start.json")
    require(
        start["plan"] == receipt(OUT / "validation-plan.json"),
        "worker start does not bind the plan",
    )
    os.chdir(ROOT)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("TNFR_TEST_MATH_BACKEND", None)
    os.environ.pop("TNFR_MATH_BACKEND", None)
    with (
        (OUT / "pytest.stdout.log").open("x", encoding="utf-8") as stdout,
        (OUT / "pytest.stderr.log").open("x", encoding="utf-8") as stderr,
    ):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            import pytest

            code = int(pytest.main(pytest_arguments()))
    verify_frozen(plan)
    write_json_once(
        OUT / "worker-result.json",
        {
            "finished_utc": utc(),
            "exit_code": code,
            "plan": receipt(OUT / "validation-plan.json"),
        },
    )
    return code


def run():
    plan = read("validation-plan.json")
    verify_baseline()
    verify_declaration()
    verify_frozen(plan)
    for name in (
        "control-start.json",
        "validation.json",
        "validation-failure.json",
        "worker-result.json",
        "control-results.xml",
        "pytest.stdout.log",
        "pytest.stderr.log",
    ):
        require(
            not (OUT / name).exists(),
            "existing evidence prohibits a second attempt: " + name,
        )
    write_json_once(
        OUT / "control-start.json",
        {
            "started_utc": utc(),
            "plan": receipt(OUT / "validation-plan.json"),
            "attempt": 1,
            "budget": BUDGET,
            "resource_scope": RESOURCE_SCOPE,
        },
    )
    job_folder = OUT / "job"
    result = None
    try:
        job_folder.mkdir(exist_ok=False)
        result = supervisor().run_supervised(
            interpreter=PYTHON,
            worker=Path(__file__),
            worker_args=("--worker",),
            folder=job_folder,
            bootstrap_path=HELPER / "bootstrap.py",
            expected_prefix=ROOT / ".venv313",
            budget=BUDGET,
            response_bytes=lambda: sum(
                path.stat().st_size
                for path in OUT.rglob("*")
                if path.is_file()
                and path.name
                not in (
                    "evaluated-source.zip",
                    "baseline.json",
                    "before-edit.zip",
                    "validation-plan.json",
                )
            ),
        )
        require(
            result["reason"] is None and result["worker_exit_code"] == 0,
            "supervised worker failed: " + str(result["reason"]),
        )
        require(
            result["cleanup_complete"] is True
            and result["job_empty_after_cleanup"] is True,
            "job cleanup incomplete",
        )
        verify_frozen(plan)
        cases = ET.parse(OUT / "control-results.xml").getroot()
        require(
            not any(
                cases.findall(".//" + tag) for tag in ("failure", "error", "skipped")
            ),
            "nonpassing JUnit result",
        )
        actual = [case.get("name") for case in cases.findall(".//testcase")]
        require(
            sorted(actual) == sorted(NAMES) and len(actual) == 6,
            "JUnit does not contain exactly the six admitted controls",
        )
        require(read("worker-result.json")["exit_code"] == 0, "worker receipt failed")
        write_json_once(
            OUT / "validation.json",
            {
                "completed_utc": utc(),
                "passed": True,
                "controls": 6,
                "plan": receipt(OUT / "validation-plan.json"),
                "test_results": receipt(OUT / "control-results.xml"),
                "stdout": receipt(OUT / "pytest.stdout.log"),
                "stderr": receipt(OUT / "pytest.stderr.log"),
                "job_result": receipt(job_folder / "supervision-result.json"),
                "job_evidence": [
                    receipt(path)
                    for path in sorted(job_folder.iterdir())
                    if path.is_file()
                ],
                "worker_result": receipt(OUT / "worker-result.json"),
                "budget": BUDGET,
                "resource_scope": RESOURCE_SCOPE,
                "resource_result": result,
                "scientific_scope": "Six exact symbolic controls of native capacity transport and additive measure; no public stage execution, trajectory, parameter scan or old producer replay.",
            },
        )
        print(
            {
                "status": "controls_passed",
                "controls": 6,
                "seconds": result["elapsed_seconds"],
                "peak_rss": result["peak_aggregate_working_set_bytes"],
            }
        )
    except BaseException as error:
        retained = [
            receipt(path)
            for path in sorted(OUT.rglob("*"))
            if path.is_file()
            and path.name
            not in (
                "validation-failure.json",
                "evaluated-source.zip",
                "baseline.json",
                "before-edit.zip",
            )
        ]
        write_json_once(
            OUT / "validation-failure.json",
            {
                "failed_utc": utc(),
                "error": f"{type(error).__name__}: {error}",
                "plan": receipt(OUT / "validation-plan.json"),
                "resource_result": result,
                "retained_outputs": retained,
                "retry_permitted": False,
            },
        )
        raise


if __name__ == "__main__":
    require(
        len(sys.argv) == 2, "select exactly --freeze, --run or acknowledged --worker"
    )
    if sys.argv[1] == "--freeze":
        freeze()
    elif sys.argv[1] == "--run":
        run()
    elif sys.argv[1] == "--worker":
        raise SystemExit(worker())
    else:
        raise ValueError("unknown validation command")
