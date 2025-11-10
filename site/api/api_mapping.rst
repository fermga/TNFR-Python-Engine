.. _api-symbol-map:

TNFR symbol → API mapping
=========================

The canonical TNFR symbols reference specific public classes and factories that
expose the mathematical objects used throughout the engine. Use this map when
you need to correlate the algebraic notation with importable Python APIs.

NFR — node factories and structural loops
-----------------------------------------

``NFR`` denotes a resonant node: its public API is exposed through the
structural helpers that seed a graph node and run validated operator
trajectories.

.. autofunction:: tnfr.structural.create_nfr

.. autofunction:: tnfr.structural.run_sequence

.. autoclass:: tnfr.node.NodeNX
   :show-inheritance:

.. doctest::

   >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
   >>> from tnfr.dynamics import set_delta_nfr_hook
   >>> from tnfr.node import NodeNX
   >>> from tnfr.structural import create_nfr, run_sequence
   >>> class StubOperator:
   ...     """Minimal operator that adjusts EPI and νf without glyph execution."""
   ...     def __init__(self, name: str, *, d_epi: float, d_vf: float) -> None:
   ...         self.name = name
   ...         self._d_epi = d_epi
   ...         self._d_vf = d_vf
   ...     def __call__(self, graph, node) -> None:
   ...         graph.nodes[node][EPI_PRIMARY] += self._d_epi
   ...         graph.nodes[node][VF_PRIMARY] += self._d_vf
   >>> graph, node = create_nfr("seed", epi=0.5, vf=1.0)
   >>> def store_delta(G, *, n_jobs=None):
   ...     nd = G.nodes[node]
   ...     nd[DNFR_PRIMARY] = nd[EPI_PRIMARY] * 0.1
   >>> set_delta_nfr_hook(graph, store_delta, note="tutorial stub")
   >>> sequence = [
   ...     StubOperator("emission", d_epi=0.1, d_vf=0.0),
   ...     StubOperator("reception", d_epi=0.05, d_vf=0.02),
   ...     StubOperator("coherence", d_epi=0.0, d_vf=0.0),
   ...     StubOperator("resonance", d_epi=0.02, d_vf=0.01),
   ...     StubOperator("silence", d_epi=0.0, d_vf=0.0),
   ... ]
   >>> run_sequence(graph, node, sequence)
   >>> adapter = NodeNX(graph, node)
   >>> round(float(adapter.EPI.f_continuous.real.max()), 3)
   0.67
   >>> round(adapter.vf, 3)
   1.03
   >>> round(graph.nodes[node][DNFR_PRIMARY], 3)
   0.067

Ĉ — coherence operator
----------------------

``Ĉ`` captures how coherence redistributes across the EPI spectrum. The
mathematical operators module provides the canonical implementation and
factories for coherence operators.

.. autoclass:: tnfr.mathematics.operators.CoherenceOperator
   :show-inheritance:

.. autofunction:: tnfr.mathematics.operators_factory.make_coherence_operator

.. doctest::

   >>> from tnfr.mathematics.operators import CoherenceOperator
   >>> from tnfr.mathematics.operators_factory import make_coherence_operator
   >>> coherence = CoherenceOperator([1.0, 0.5, 0.25])
   >>> round(coherence.spectral_bandwidth(), 2)
   0.75
   >>> round(coherence.expectation([1 + 0j, 1 + 0j, 0j]), 2)
   0.75
   >>> fabricated = make_coherence_operator(2, c_min=0.3)
   >>> fabricated.spectrum().real.tolist()
   [0.3, 0.3]

Ĵ — frequency operator
----------------------

``Ĵ`` is the structural frequency operator paired with ``Ĉ``. It ensures
νf remains non-negative and aligned with the Hilbert basis.

.. autoclass:: tnfr.mathematics.operators.FrequencyOperator
   :show-inheritance:

.. autofunction:: tnfr.mathematics.operators_factory.make_frequency_operator

.. doctest::

   >>> import numpy as np
   >>> from tnfr.mathematics.operators import FrequencyOperator
   >>> from tnfr.mathematics.operators_factory import make_frequency_operator
   >>> matrix = np.array([[1.0, 0.2], [0.2, 0.5]], dtype=float)
   >>> frequency = FrequencyOperator(matrix)
   >>> round(frequency.project_frequency([1 + 0j, 0j]), 2)
   1.0
   >>> canonical = make_frequency_operator(np.array([[0.6, 0.0], [0.0, 0.4]], dtype=float))
   >>> canonical.spectrum().tolist()
   [0.4, 0.6]

ΔNFR — canonical reorganisation hook
------------------------------------

``ΔNFR`` measures the internal demand for reorganisation. The structural layer
installs a canonical hook that mixes EPI and νf before every operator step.

.. autofunction:: tnfr.dynamics.dnfr.dnfr_epi_vf_mixed

.. doctest::

   >>> from tnfr.constants import DNFR_PRIMARY
   >>> from tnfr.structural import create_nfr
   >>> from tnfr.dynamics.dnfr import dnfr_epi_vf_mixed
   >>> graph, left = create_nfr("left", epi=1.0, vf=0.5)
   >>> _, right = create_nfr("right", epi=2.0, vf=1.2, graph=graph)
   >>> graph.add_edge(left, right)
   >>> dnfr_epi_vf_mixed(graph)
   >>> [round(graph.nodes[n][DNFR_PRIMARY], 3) for n in (left, right)]
   [0.85, -0.85]
   >>> graph.graph["_DNFR_META"]["hook"]
   'dnfr_epi_vf_mixed'

EPI — primary information structure
-----------------------------------

``EPI`` stores the coherent form that operators reorganise. The mathematics
package exposes Banach elements validated against the canonical domain.

.. autoclass:: tnfr.mathematics.epi.BEPIElement
   :show-inheritance:

.. autoclass:: tnfr.mathematics.spaces.BanachSpaceEPI
   :show-inheritance:
.. seealso::

   {doc}`api/canonical_validators` for runtime validation helpers.

.. doctest::

   >>> import numpy as np
   >>> from tnfr.mathematics.epi import BEPIElement
   >>> from tnfr.mathematics.spaces import BanachSpaceEPI, HilbertSpace
   >>> grid = np.linspace(0.0, 1.0, 3)
   >>> element = BEPIElement([1 + 0j, 0.5 + 0j, 0j], [0.2 + 0j, 0.1 + 0j], grid)
   >>> round(float(element.adjoint()), 2)
   1.0
   >>> space = BanachSpaceEPI()
   >>> zero = space.zero_element(continuous_size=3, discrete_size=2)
   >>> combined = space.direct_sum(element, zero)
   >>> round(float(combined), 2)
   1.0
   >>> basis = space.canonical_basis(continuous_size=3, discrete_size=2, continuous_index=1, discrete_index=0)
   >>> hilbert = HilbertSpace(dimension=2)
   >>> space.tensor_with_hilbert(basis, hilbert).real.tolist()
   [[1.0, 0.0], [0.0, 0.0]]
