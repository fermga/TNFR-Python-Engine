from __future__ import annotations

import os

from examples import particle_atlas, atom_atlas, molecule_atlas, triatomic_atlas, periodic_table_atlas


def _assert_panel_in_html(html_path: str) -> None:
    assert os.path.exists(html_path), f"missing html: {html_path}"
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    # Check key Safety Triad panel markers
    assert "Safety Triad (telemetry-only)" in html
    assert "ΔΦ_s threshold:" in html
    assert "|∇φ| threshold:" in html
    assert "|K_φ| threshold:" in html
    assert "U6 is descriptive only; no control feedback into dynamics." in html


def test_safety_triad_panel_particle_atlas():
    html_path = particle_atlas.main()
    _assert_panel_in_html(html_path)


def test_safety_triad_panel_atom_atlas():
    html_path = atom_atlas.main()
    _assert_panel_in_html(html_path)


def test_safety_triad_panel_molecule_atlas():
    html_path = molecule_atlas.main()
    _assert_panel_in_html(html_path)


def test_safety_triad_panel_triatomic_atlas():
    html_path = triatomic_atlas.main()
    _assert_panel_in_html(html_path)


def test_safety_triad_panel_periodic_table_atlas():
    html_path = periodic_table_atlas.main()
    _assert_panel_in_html(html_path)
