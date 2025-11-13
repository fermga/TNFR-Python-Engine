from __future__ import annotations

from tnfr.examples_utils.html import render_safety_triad_panel


def test_render_safety_triad_panel_contains_thresholds_and_disclaimer():
    html = render_safety_triad_panel(
        thresholds={
            "phi_delta": 2.0,
            "grad": 0.38,
            "kphi": 3.0,
        }
    )

    # Threshold labels
    assert "Safety Triad (telemetry-only)" in html
    assert "ΔΦ_s threshold: 2.0" in html
    assert "|∇φ| threshold: 0.38" in html
    assert "|K_φ| threshold: 3.0" in html

    # Telemetry-only disclaimer
    assert "U6 is descriptive only; no control feedback into dynamics." in html
