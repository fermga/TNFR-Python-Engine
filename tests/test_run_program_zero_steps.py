from argparse import Namespace
from tnfr.cli.execution import run_program
from tnfr.glyph_history import ensure_history


def test_run_program_respects_zero_steps():
    args = Namespace(
        nodes=3,
        topology="ring",
        seed=1,
        p=None,
        config=None,
        dt=None,
        integrator=None,
        remesh_mode=None,
        glyph_hysteresis_window=None,
        grammar_canon=True,
        selector="basic",
        gamma_type=None,
        gamma_beta=None,
        gamma_R0=None,
        observer=False,
        save_history=None,
        export_history_base=None,
        export_format="json",
        steps=0,
    )
    G = run_program(None, None, args)
    hist = ensure_history(G)
    assert len(hist.get("C_steps", [])) == 1

