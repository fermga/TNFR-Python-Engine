.. _api-canonical-validators:

Canonical validators and dynamics checks
=======================================

Use these validators to keep simulations within the canonical TNFR
constraints. Each entry links to the public API and includes a minimal doctest
that runs in CI.

stable_unitary — unitary stability check
----------------------------------------

.. autofunction:: tnfr.mathematics.runtime.stable_unitary

Example
~~~~~~~

.. doctest::
   >>> import numpy as np
   >>> from tnfr.mathematics.operators import CoherenceOperator
   >>> from tnfr.mathematics.runtime import stable_unitary
   >>> from tnfr.mathematics.spaces import HilbertSpace
   >>> hilbert = HilbertSpace(2)
   >>> operator = CoherenceOperator([0.0, 0.5])
   >>> state = np.array([1.0, 0.0], dtype=np.complex128)
   >>> passed, norm_after = stable_unitary(state, operator, hilbert)
   >>> passed
   True
   >>> round(norm_after, 6)
   1.0

frequency_positive — structural frequency guard
-----------------------------------------------

.. autofunction:: tnfr.mathematics.runtime.frequency_positive

Example
~~~~~~~

.. doctest::
   >>> import numpy as np
   >>> from tnfr.mathematics.operators import FrequencyOperator
   >>> from tnfr.mathematics.runtime import frequency_positive
   >>> operator = FrequencyOperator(np.diag([0.2, 0.4]))
   >>> summary = frequency_positive(np.array([1.0, 0.0], dtype=np.complex128), operator)
   >>> summary["passed"], round(summary["value"], 3)
   (True, 0.2)

ContractiveDynamicsEngine — dissipative evolution monitor
---------------------------------------------------------

.. autoclass:: tnfr.mathematics.dynamics.ContractiveDynamicsEngine
   :show-inheritance:

Example
~~~~~~~

.. doctest::
   >>> import numpy as np
   >>> from tnfr.mathematics.dynamics import ContractiveDynamicsEngine
   >>> from tnfr.mathematics.spaces import HilbertSpace
   >>> hilbert = HilbertSpace(2)
   >>> lindblad = -0.05 * np.eye(4, dtype=np.complex128)
   >>> engine = ContractiveDynamicsEngine(lindblad, hilbert)
   >>> rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
   >>> rho1 = engine.step(rho0, dt=0.25)
   >>> float(np.trace(rho1).real)
   1.0
   >>> engine.last_contractivity_gap >= -1e-9
   True
