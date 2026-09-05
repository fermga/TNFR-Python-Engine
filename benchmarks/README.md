# TNFR benchmark guide

The scripts in this directory are research and performance instruments. Their
outputs are evidence only for the recorded code revision, environment, inputs and
seed. A benchmark result does not promote a heuristic to a theorem or establish a
universal speedup.

## Running a benchmark

Install the repository in editable mode, then invoke the selected script:

```bash
python -m pip install -e ".[dev-minimal]"
python benchmarks/riemann_program.py
python benchmarks/directed_nonnormal_dynamics.py
```

Use `--help` when a script exposes command-line options. Optional numerical
backends may be skipped when their dependencies are absent.

## Main groups

| Group | Representative instruments |
| --- | --- |
| Core dynamics | `conservation_law_validation.py`, `field_methods_battery.py`, `u2_destabilization_irreversibility.py` |
| Directed transport | `directed_nonnormal_dynamics.py`, `directed_transient_u2.py`, `heterogeneous_vf_boundary.py` |
| Arithmetic structure | `arithmetic_pressure_audit.py`, `arithmetic_pulse_recurrence.py`, `crt_multiscale_composition.py` |
| Riemann research | `riemann_program.py`, `remesh_infinity_riemann_*.py`, `phase_wall.py` |
| Structural interfaces | `structural_interface_benchmark.py`, `temporal_interface_benchmark.py`, `multichannel_interface_benchmark.py` |
| Emergent-model probes | files beginning with `emergent_`; interpret them under their explicit model assumptions |

The filename and module docstring are the current description of each
instrument. Research conclusions belong in the corresponding document under
[theory/](../theory/README.md), where exact, measured, negative and open results
are separated.

## Required reporting

A reusable benchmark record must include:

1. repository revision and Python/dependency versions;
2. operating system and numerical backend;
3. input domain, graph construction and seed;
4. operator sequence and configuration overrides;
5. C(t), Si, phase, `nu_f` and relevant tetrad telemetry;
6. raw result artifact or checksum;
7. limitations and any circular input, known factors or labels.

Compare implementations on identical inputs and warm-up policy. Report absolute
times alongside ratios and never copy a machine-specific sample result into the
canonical API documentation.
