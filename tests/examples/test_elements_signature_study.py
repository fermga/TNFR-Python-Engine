from __future__ import annotations

import os

from examples.elements_signature_study import main as run_elements


def test_elements_signature_study_outputs(tmp_path):
    out_html = run_elements()
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "output")
    base_dir = os.path.abspath(base_dir)

    html_path = out_html
    csv_path = os.path.join(base_dir, "elements_signature_study.csv")
    jsonl_path = os.path.join(base_dir, "elements_signature_study.jsonl")

    assert os.path.exists(html_path), f"missing html output: {html_path}"
    assert os.path.exists(csv_path), f"missing csv output: {csv_path}"
    assert os.path.exists(jsonl_path), f"missing jsonl output: {jsonl_path}"
