"""Portable integration and report output; no retained producer is executed."""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import sys

import pytest

from benchmarks import thol_regional_response_criterion as study
from tests.physics.test_thol_child_distortion_audit import _fixture


def _admitted():
    left, right, reference, children = _fixture(defects=True)
    reader = study.admission.child
    pair = study.admission._pair(reader._admit_reset(left, reference), reader._admit_reset(right, reference))
    s, a = pair["control"]["S"], pair["control"]["A"]
    t = tuple(tuple(x-F(1, 4)*y for x, y in zip(sr, ar, strict=True)) for sr, ar in zip(s, a, strict=True))
    return {"original_reference": reference, "children": children,
            "common_coefficients": {"S": s, "A": a, "T": t},
            "witnesses": {"synthetic": pair}}


def test_real_manifest_and_complete_dataclass_report_are_serializable(tmp_path, monkeypatch):
    admitted = _admitted()
    retained = tmp_path/"retained.json"
    retained.write_text("{}", encoding="utf-8")
    admitted["historical_inputs"] = {"synthetic": {
        "path": str(retained), "sha256": hashlib.sha256(retained.read_bytes()).hexdigest()}}
    monkeypatch.setattr(study.admission, "load_comparable_witnesses", lambda *a, **k: admitted)
    monkeypatch.setattr(study, "current_git_source_provenance", lambda *a: ("a"*40, False, None))
    output = tmp_path/"criterion.json"
    monkeypatch.setattr(sys, "argv", ["criterion", "--output", str(output)])
    study.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["manifest"]["claim_id"] == "O3.a-conditional-regional-response"
    assert result["admission"]["original_reference"]["metric_weights"]
    assert result["admission"]["witnesses"]["synthetic"]["control"]["observation"]["snapshot"]
    assert result["native_calls"] == result["kernel_calls"] == result["new_trajectories"] == 0
    assert not result["prospective_binary64_bound_certified"]


@pytest.mark.parametrize("mutate", ("endpoint", "source"))
def test_full_endpoint_and_paired_source_admission_are_required(mutate):
    admitted = deepcopy(_admitted())
    witness = admitted["witnesses"]["synthetic"]
    if mutate == "endpoint":
        witness["delta"]["xf"] = tuple(x+1 for x in witness["delta"]["xf"])
    else:
        witness["paired_source_difference"] = (F(1),)*4
    with pytest.raises(ValueError):
        study.analyze_admitted(admitted)


def test_input_overwrite_refused_before_reading(tmp_path, monkeypatch):
    path = tmp_path/"input.json"
    monkeypatch.setattr(sys, "argv", ["criterion", "--reset-input", str(path), "--output", str(path)])

    def forbidden(*args, **kwargs):
        raise AssertionError("must refuse before input load")
    monkeypatch.setattr(study.admission, "load_comparable_witnesses", forbidden)
    with pytest.raises(ValueError):
        study.main()
