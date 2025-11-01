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

Ĉ — coherence operator
----------------------

``Ĉ`` captures how coherence redistributes across the EPI spectrum. The
mathematical operators module provides the canonical implementation and
factories for coherence operators.

.. autoclass:: tnfr.mathematics.operators.CoherenceOperator
   :show-inheritance:

.. autofunction:: tnfr.mathematics.operators_factory.make_coherence_operator

Ĵ — frequency operator
----------------------

``Ĵ`` is the structural frequency operator paired with ``Ĉ``. It ensures
νf remains non-negative and aligned with the Hilbert basis.

.. autoclass:: tnfr.mathematics.operators.FrequencyOperator
   :show-inheritance:

.. autofunction:: tnfr.mathematics.operators_factory.make_frequency_operator

ΔNFR — canonical reorganisation hook
------------------------------------

``ΔNFR`` measures the internal demand for reorganisation. The structural layer
installs a canonical hook that mixes EPI and νf before every operator step.

.. autofunction:: tnfr.dynamics.dnfr.dnfr_epi_vf_mixed

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
