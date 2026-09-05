# TNFR Test Suite

[TESTING.md](../TESTING.md) is the authoritative guide for installation,
commands, markers, backend selection, structural regression expectations and
validation reporting. Test physics and contracts follow [AGENTS.md](../AGENTS.md).

| Directory | Scope |
|---|---|
| [core_physics/](core_physics/) | Nodal equation, triad and pressure channels |
| [operators/](operators/) | Operators, contracts and grammar |
| [physics/](physics/) | Fields, diffusion, symmetry, conservation and caches |
| [mathematics/](mathematics/) | Backends, arithmetic networks and multiscale constructions |
| [sdk/](sdk/) | Public network interface |
| [engines/](engines/) | Self-optimization and discovery manifests |
| [parallel/](parallel/) | Partition manifests |
| [research/](research/) | Research infrastructure |
| [scripts/](scripts/) | Script integration |
| [data/](data/) | Fixture data and manifests |

Top-level test files cover phase-gate and external data interfaces, replay,
distributed FFT, factorization and mechanics. [conftest.py](conftest.py) and
[utils.py](utils.py) provide shared infrastructure.
