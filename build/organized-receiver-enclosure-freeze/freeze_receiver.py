"""Freeze one receiver enclosure without executing the reserved trajectory."""

import importlib.util
import subprocess
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.mathematics._rational_interval import INTERVAL_METHOD  # noqa: E402
from tnfr.research.artifact_io import (  # noqa: E402
    file_receipt,
    sha256_bytes,
    write_json_once,
)
from tnfr.research.frozen_source import inspect_frozen_source  # noqa: E402
from tnfr.research.organized_receiver_enclosure import (  # noqa: E402
    receiver_enclosure_policy,
    receiver_enclosure_source,
)
from tnfr.utils.io import json_dumps  # noqa: E402

STEM = "docs/assets/sine_formed_classes/organized-receiver-enclosure-v1"
EVALUATOR = "build/organized-receiver-enclosure-freeze/evaluate_receiver.py"
FREEZER = "build/organized-receiver-enclosure-freeze/freeze_receiver.py"
PROOF = "theory/nodal/ORGANIZED_RECEIVER_ENCLOSURE.md"


def main():
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(["git", "diff", "--quiet", base, "--", "src"], cwd=ROOT, check=True)
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src"], cwd=ROOT
    ).strip():
        raise ValueError("commit the complete runtime before freezing")
    for suffix in (
        ".protocol.json",
        ".source.zip",
        ".freeze.json",
        ".attempt.json",
        ".json",
        ".export-error.json",
        ".steps",
    ):
        if (ROOT / (STEM + suffix)).exists():
            raise FileExistsError("preserve the first freeze, attempt and prefix")
    specification = importlib.util.spec_from_file_location(
        "future_receiver_evaluator", ROOT / EVALUATOR
    )
    evaluator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(evaluator)  # Definitions only; never main.
    policy = receiver_enclosure_policy()
    admission = asdict(receiver_enclosure_source())
    protocol = {
        "schema": "tnfr.organized-receiver-enclosure-protocol.v1",
        "evaluation_status_at_freeze": "not_evaluated",
        "source_base_commit": base,
        "runtime_overlays": [],
        "runtime_environment": evaluator._runtime_environment(),
        "theory_owner": PROOF + "#organized-receiver-enclosure-protocol",
        "source_specification": {
            "prior_artifact_receipts": [],
            "preparation_owner": "theory/nodal/ORGANIZED_RECEIVER_MECHANISM.md#organized-receiver-source",
            "admission_owner": "theory/nodal/ORGANIZED_RECEIVER_SLOW_ADMISSION.md#organized-receiver-capture-contract",
            "retained_source": "original two-ramp source; independent form and lifted-phase errors bounded by 1e-10 with separate exact zero sums in each original C9",
            "nominal_initial_form": "m*(j-4) in regions 0,1 and zero in region 2; m=(2046/9)*(355/113)**2",
            "nominal_initial_phase": "zero at every node in the supplied common continuous lift",
            "node_order": tuple(range(27)),
            "node_label": "node=9*r+j; r=0,1,2; j=0,...,8",
            "state_order": "x[0:27], theta[0:27]; nominal reference uses 13 reflected coordinates, actual uncertainty retains the complete mean-zero leaf",
            "support": "three unit C9 cycles; oriented region contacts (0,1),(1,2),(2,0), offsets 0,1 with weight 1/2; weighted degree four; present from preparation",
            "capacities": (Q(1),) * 27,
            "clock": "tau=(1023/1024)*t; u=gamma**2*(tau-400); gamma=1/(1023*pi); both rows transformed",
            "form_row": "x_tau=-L*x/4+gamma*S(theta)/4",
            "phase_row": "theta_tau=gamma*L*x/4",
            "pressure_refresh": "continuous state-dependent weighted sine law",
            "forcing": "none; no Gamma source, state resets, support events or conductance evolution",
            "conserved_means": "only global form and continuous-phase means after connected evolution starts",
            "source_cost": "complete weighted initial storage, including contacts; no isolated recovered-source certificate",
            "release": "source-inclusive phase ball and carried original form from tau=400; donor retention from tau=200 to release independently admitted",
            "seed": None,
            "interval_method": INTERVAL_METHOD,
            "physical_identification": "not supplied or tested",
        },
        "producer": "tnfr.research.organized_receiver_enclosure.run_receiver_enclosure",
        "numerical_policy": policy,
        "kernel_policy": {
            "picard_iterations": 16,
            "picard_radius_multiplier": Q(5, 4),
            "picard_additive_padding": Q(1, 2**90),
            "metric": "2*I_13; exact reflection lift preserves the full Euclidean norm",
            "interval_grid_bits": 128,
            "source_and_remainder": "full-source phase radius plus complete-row defect; local Taylor remainder added at every endpoint",
        },
        "source_admission": admission,
        "reserved_attempt_policy": {
            "maximum_attempts": 1,
            "maximum_step_records": 256,
            "adaptation_or_retry": False,
            "alternative_route": False,
            "preserve_incomplete_response": True,
            "step_retention": "exclusive sequential JSON files for accepted steps and any failed attempt, written before the next step; final outcome retains exact byte receipts",
            "interruption": "preserve exclusive attempt and every retained prefix file; no automatic resume or retry",
            "export_failure": "preserve partial output and prefix; record export-error metadata when possible",
            "positive_outcome": "whole-source donor retention and same-state receiver winding one, full target cell and full form-phase capture",
            "donor_loss_outcome": "at least one donor has robust endpoint winding different from one; quantified for actual admitted source through the full tube",
            "unavailable_outcome": "enclosure, uncertainty or horizon budget exhausted without the required certificate; not proof of nonformation",
        },
    }
    write_json_once(ROOT / (STEM + ".protocol.json"), protocol)
    names = [
        STEM + ".protocol.json",
        EVALUATOR,
        FREEZER,
        PROOF,
        "theory/nodal/ORGANIZED_RECEIVER_SLOW_ADMISSION.md",
        "theory/nodal/ORGANIZED_RECEIVER_MECHANISM.md",
        "theory/nodal/ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md",
        "src/tnfr/research/organized_receiver_enclosure.py",
        "src/tnfr/research/artifact_io.py",
        "src/tnfr/research/frozen_source.py",
        "src/tnfr/physics/_organized_receiver_enclosure_geometry.py",
        "src/tnfr/physics/_organized_receiver_slow_admission.py",
        "src/tnfr/physics/_organized_receiver_mechanism.py",
        "src/tnfr/physics/_organized_network_capture.py",
        "src/tnfr/physics/_organized_network_compatibility.py",
        "src/tnfr/physics/_sine_preparation.py",
        "src/tnfr/physics/reversible_eigenmode_reference.py",
        "src/tnfr/physics/relational_sine_formed_classes.py",
        "src/tnfr/mathematics/_validated_metric.py",
        "src/tnfr/mathematics/_comparison_flow.py",
        "src/tnfr/mathematics/_exact_linear_algebra.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/mathematics/_interval_taylor.py",
        "src/tnfr/mathematics/_rational_interval.py",
        "src/tnfr/mathematics/_phase_midpoint.py",
        "src/tnfr/utils/io.py",
        "src/tnfr/_exact_time.py",
        "pyproject.toml",
    ]
    retained = {name: (ROOT / name).read_bytes() for name in names}
    files = []
    for name, data in retained.items():
        item = {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}
        if name.startswith("src/"):
            committed = subprocess.check_output(
                ["git", "show", f"{base}:{name}"], cwd=ROOT
            )
            normalized = data.replace(b"\r\n", b"\n")
            if normalized != committed.replace(b"\r\n", b"\n"):
                raise ValueError(f"runtime snapshot differs: {name}")
            item.update(
                git_blob_sha256=sha256_bytes(committed),
                normalized_lf_sha256=sha256_bytes(normalized),
            )
        files.append(item)
    manifest = {
        "schema": "tnfr.organized-receiver-enclosure-source-snapshot.v1",
        "source_base_commit": base,
        "runtime_overlays": [],
        "files": files,
    }
    with zipfile.ZipFile(
        ROOT / (STEM + ".source.zip"), "x", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("source-manifest.json", json_dumps(manifest, indent=2) + "\n")
        for name, data in retained.items():
            archive.writestr(name, data)
    receipt = {
        "schema": "tnfr.organized-receiver-enclosure-freeze.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status_at_freeze": "not_evaluated",
        "source_base_commit": base,
        "runtime_overlays": [],
        "future_evaluator": EVALUATOR,
        "artifacts": [
            file_receipt(ROOT / (STEM + suffix), label=STEM + suffix)
            for suffix in (".protocol.json", ".source.zip")
        ],
        "claim": "prospective byte and complete Git-base association; not authenticated chronology, execution or physical validation",
    }
    write_json_once(ROOT / (STEM + ".freeze.json"), receipt)
    print(inspect_frozen_source(ROOT, STEM + ".freeze.json"))


if __name__ == "__main__":
    main()
