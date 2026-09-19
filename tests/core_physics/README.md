# Core physics test scope

Tests here exercise the implemented nodal pressure, integration and diagnostic
contracts. The suite does not establish the ontology or dimensional completeness
of TNFR. [TESTING.md](../../TESTING.md) owns execution instructions.

## Retired self-contained illustrations

The 2026-09-19 audit removed `test_nodal_equation.py` and
`test_structural_triad.py` (11 tests each). Most cases assigned strings or numbers
to NetworkX attributes and asserted those same values or their ordinary Python
products. They did not call the TNFR form integrator, scalar-domain validator or
coupling operator. Such checks cannot establish Banach structure, fractality,
operator-only evolution, a special golden-ratio capacity, or an Euler-constant
stability bound. Zero capacity suppresses the unforced form channel; it is not a
general node-death theorem.

The one production field call was a finite-potential smoke check on a complete
graph. Existing tests exercise that reader with stronger independent numerical
oracles. No scientific producer or current implementation was removed.

| Intended contract | Retained executable owner |
| --- | --- |
| Capacity multiplies pressure once; zero capacity suppresses the unforced rate | [Stable neighbor pressure](test_stable_neighbor_pressure.py), particularly `test_nodal_integration_applies_capacity_once_and_zero_capacity_freezes` |
| Optional forcing remains a distinct model at zero capacity | [Nodal forcing scope](../test_nodal_forcing_scope.py) |
| Represented nodal updates and retained remainder encoding | [Nodal remainder kernel](../test_nodal_remainder_kernel.py) |
| Signed EPI, serialized scalar embeddings and richer-form rejection | [EPI scalarization](../physics/test_epi_scalarization.py), [operator EPI domain](../operators/test_epi_domain_consistency.py) |
| Circular phase admission and mutation atomicity | [U3 hard invariant](../operators/test_u3_hard_invariant.py) |
| Structural potential on declared distances and pressure | [Potential accuracy](../physics/test_structural_potential_accuracy.py) |
| Actual nested structure creation and protected failure paths | [Self-organization atomicity](../operators/test_self_organization_atomicity.py) |

Original bytes and SHA-256 receipts are preserved locally under
`artifacts/release/v0.0.3.6/obsolete_core_test_originals/`. Git history retains the
removed tracked files. These illustrations are historical material, not active
acceptance criteria. This mapping records why each intended contract is retained
without keeping tautological tests.
