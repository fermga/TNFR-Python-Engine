import pytest

from tnfr.constants import (
    EPI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Coherence, Dissonance, Silence
from tnfr.dynamics.runtime import step


def _scalar(x):
    """Extract a scalar float from possible TNFR rich structures."""
    if isinstance(x, dict):
        # Common pattern: {'continuous': value or (value, ...), ...}
        if "continuous" in x:
            v = x["continuous"]
            if isinstance(v, (list, tuple)):
                v = v[0] if v else 0.0
            if isinstance(v, complex):
                return float(abs(v))
            return float(v)
        # Fallback: try any numeric-looking field
        for k in ("value", "mean", "avg"):
            if k in x:
                return float(x[k])
        return 0.0
    if isinstance(x, complex):
        return float(abs(x))
    return float(x)


def _state_tuple(G, n):
    return (
        _scalar(G.nodes[n][EPI_PRIMARY]),
        _scalar(G.nodes[n][VF_PRIMARY]),
        _scalar(G.nodes[n][THETA_PRIMARY]),
    )


def test_reproducibility_same_initial_conditions():
    """
    Same initial graph and same sequence must produce identical
    trajectories (determinism).
    """
    G1, n1 = create_nfr("seed1", epi=1.0, vf=1.5, theta=0.1)
    G2, n2 = create_nfr("seed1", epi=1.0, vf=1.5, theta=0.1)

    seq = [Emission(), Coherence(), Silence()]
    run_sequence(G1, n1, seq)
    run_sequence(G2, n2, seq)

    s1 = _state_tuple(G1, n1)
    s2 = _state_tuple(G2, n2)
    assert s1 == pytest.approx(s2, rel=1e-9, abs=1e-12)


def test_sequence_requires_closure_U1b():
    """
    Sequences must end with a closure (U1b);
    missing closure should fail validation.
    """
    G, n = create_nfr("node-u1b", epi=0.8, vf=1.0, theta=0.0)

    with pytest.raises(ValueError):
        run_sequence(G, n, [Emission(), Coherence()])  # no closure

    # Valid sequence with closure should not raise
    run_sequence(G, n, [Emission(), Coherence(), Silence()])


def test_stabilizer_after_destabilizer_U2():
    """
    Destabilizers (e.g., Dissonance) require stabilizers
    (Coherence/SelfOrganization) afterwards (U2).
    """
    G, n = create_nfr("node-u2", epi=0.5, vf=1.2, theta=0.0)

    # Invalid: destabilizer followed by closure without stabilizer
    with pytest.raises(ValueError):
        run_sequence(G, n, [Emission(), Dissonance(), Silence()])

    # Valid: include stabilizer after destabilizer
    run_sequence(G, n, [Emission(), Dissonance(), Coherence(), Silence()])


def test_silence_latency_invariance():
    """Silence keeps EPI invariant across a dynamics step (latency)."""
    G, n = create_nfr("node-sha", epi=0.8, vf=1.0, theta=0.0)
    # Enter silent state coherently (respect U1a/U1b and U2)
    run_sequence(G, n, [Emission(), Coherence(), Silence()])
    # Record EPI right after entering Silence
    before = _scalar(G.nodes[n][EPI_PRIMARY])
    # Advance one step; Silence should freeze evolution (νf→0 ⇒ ΔNFR→0)
    inject_defaults(G)
    step(G, dt=1.0, use_Si=False, apply_glyphs=False)

    after = _scalar(G.nodes[n][EPI_PRIMARY])
    assert after == pytest.approx(before, rel=0, abs=1e-12)
