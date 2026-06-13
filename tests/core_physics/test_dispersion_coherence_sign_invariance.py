"""Regression tests for the dispersion-based auxiliary coherence.

Guards against a sign-asymmetry bug in which
:func:`tnfr.metrics.coherence.compute_global_coherence` and
:func:`tnfr.metrics.coherence.compute_local_coherence` normalized the ΔNFR
standard deviation by ``max(ΔNFR)`` instead of ``max|ΔNFR|``. With an
all-negative ΔNFR field the raw maximum is the *least* negative value, so
``σ / max`` turned negative and ``C_disp = 1 − σ/max`` exceeded 1.0 and was
clipped to 1.0 — reporting *perfect* coherence for a field that is in fact
highly dispersed. The diagnostic is meant to be invariant under the global
sign of structural pressure (it is a homogeneity probe), so the mirror
fields ``+v`` and ``−v`` must yield identical coherence.

No existing test exercised these two functions, which is why the bug went
undetected; hence these explicit sign-symmetry, scale-invariance and
magnitude-normalization checks.
"""
from __future__ import annotations

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.metrics.coherence import (
    compute_global_coherence,
    compute_local_coherence,
)


def _path_graph(values: list[float]) -> nx.Graph:
    """Path graph 1-2-3-4 with the given ΔNFR values assigned in order."""
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    for node, value in zip(G.nodes(), values):
        set_attr(G.nodes[node], ALIAS_DNFR, value)
    return G


class TestDispersionCoherenceSignInvariance:
    """C_disp must depend only on ΔNFR *magnitude* dispersion, not its sign."""

    def test_global_coherence_sign_symmetric(self) -> None:
        """Mirror fields ``+v`` and ``−v`` must give identical global C_disp.

        Regression: all-negative ΔNFR previously normalized by the raw
        (least-negative) maximum, yielding C_disp = 1.0 (false "perfect
        coherence") for a dispersed field.
        """
        positive = _path_graph([0.1, 0.2, 0.3, 0.4])
        negative = _path_graph([-0.1, -0.2, -0.3, -0.4])

        c_pos = compute_global_coherence(positive)
        c_neg = compute_global_coherence(negative)

        assert c_neg == pytest.approx(c_pos, abs=1e-12)
        # And it must NOT be the falsely-perfect 1.0 of the old bug.
        assert c_neg < 1.0
        assert 0.0 <= c_neg <= 1.0

    def test_local_coherence_sign_symmetric(self) -> None:
        """Mirror fields must give identical local C_disp at the same node."""
        positive = _path_graph([0.1, 0.2, 0.3, 0.4])
        negative = _path_graph([-0.1, -0.2, -0.3, -0.4])

        c_pos = compute_local_coherence(positive, node=2, radius=1)
        c_neg = compute_local_coherence(negative, node=2, radius=1)

        assert c_neg == pytest.approx(c_pos, abs=1e-12)
        assert c_neg < 1.0
        assert 0.0 <= c_neg <= 1.0

    def test_global_coherence_scale_invariant(self) -> None:
        """C_disp is invariant under proportional scaling of ΔNFR."""
        base = _path_graph([0.1, 0.2, 0.3, 0.4])
        scaled = _path_graph([1.0, 2.0, 3.0, 4.0])

        assert compute_global_coherence(scaled) == pytest.approx(
            compute_global_coherence(base), abs=1e-12
        )

    def test_global_coherence_normalized_by_magnitude(self) -> None:
        """Mixed-sign field normalizes by max|ΔNFR|, staying within (0, 1).

        With values [0.1, −0.5, 0.3, −0.2] the magnitude scale is
        max|ΔNFR| = 0.5; the result must be a proper interior coherence,
        never clipped to the degenerate endpoints.
        """
        mixed = _path_graph([0.1, -0.5, 0.3, -0.2])
        c_mixed = compute_global_coherence(mixed)

        assert 0.0 < c_mixed < 1.0

    def test_uniform_field_is_perfectly_coherent(self) -> None:
        """A spatially uniform ΔNFR field has zero dispersion ⇒ C_disp = 1."""
        uniform = _path_graph([0.3, 0.3, 0.3, 0.3])
        assert compute_global_coherence(uniform) == pytest.approx(1.0, abs=1e-12)

    def test_uniform_negative_field_is_perfectly_coherent(self) -> None:
        """A uniform *negative* field is equally coherent (sign-invariant)."""
        uniform_neg = _path_graph([-0.3, -0.3, -0.3, -0.3])
        assert compute_global_coherence(uniform_neg) == pytest.approx(
            1.0, abs=1e-12
        )
