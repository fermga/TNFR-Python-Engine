# Core physics test scope

Tests here exercise the implemented nodal pressure, integration and diagnostic
contracts. The suite does not establish the ontology or dimensional completeness
of TNFR. [TESTING.md](../../TESTING.md) owns execution instructions.

Balance tests exercise divergence, temporal secants, sector decomposition and
operator-to-tracker wiring. The P3 counterexample distinguishes an unweighted
sum from degree-weighted cancellation. Role/registry checks verify configured
policies; they do not derive energy bounds. Zero capacity suppresses the
unforced form row, not every possible evolution channel.

## Contract coverage

| Intended contract | Retained executable owner |
| --- | --- |
| Capacity multiplies pressure once; zero capacity suppresses the unforced rate | [Stable neighbor pressure](test_stable_neighbor_pressure.py), particularly `test_nodal_integration_applies_capacity_once_and_zero_capacity_freezes` |
| Optional forcing remains a distinct model at zero capacity | [Nodal forcing scope](../test_nodal_forcing_scope.py) |
| Represented nodal updates and retained remainder encoding | [Nodal remainder kernel](../test_nodal_remainder_kernel.py) |
| Finite capacity/pressure admission, advancing substep clocks and solver-output failure restoration | [Integrator numerics](../test_integrator_numerics.py) |
| Signed EPI, serialized scalar embeddings and richer-form rejection | [EPI scalarization](../physics/test_epi_scalarization.py), [operator EPI domain](../operators/test_epi_domain_consistency.py) |
| Form-chart equivalence and capacity identifiability under a fixed pressure law | [Nodal foundations](../physics/test_nodal_foundation_scope.py), [capacity scope](../physics/test_constitutive_capacity_scope.py) |
| Pressure scalar admission, live public weights and optional model boundaries | [Pressure read contract](test_pressure_read_contract.py), [phase path controls](../test_dnfr_fallback_parity.py), [constitutive scope](../physics/test_pressure_constitutive_scope.py) |
| Circular phase admission and mutation atomicity | [U3 hard invariant](../operators/test_u3_hard_invariant.py) |
| Structural potential on declared distances and pressure | [Potential accuracy](../physics/test_structural_potential_accuracy.py) |
| Actual nested structure creation and protected failure paths | [Self-organization atomicity](../operators/test_self_organization_atomicity.py) |
