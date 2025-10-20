# Examples

Runnable examples illustrate how the TNFR engine orchestrates canonical operators in
realistic scenarios. Each script can be executed directly with Python 3.9+ after installing
`tnfr` and optional extras from the [quickstart](../getting-started/quickstart.md).

## Controlled dissonance with re-coherence

- Script: [`controlled_dissonance.py`](controlled_dissonance.py)
- Summary: three-node ring where node C receives a controlled dissonance pulse, bifurcates, and
  re-stabilises while telemetry records C(t), ΔNFR, Si, and history windows.
- Run:

  ```bash
  python docs/examples/controlled_dissonance.py
  ```

## Optical cavity feedback loop

- Script: [`optical_cavity_feedback.py`](optical_cavity_feedback.py)
- Summary: tabletop optical cavity (laser head, piezo mirror stage, detector array) realigns
  after a thermal drift using self-organisation, mutation, and resonance sequences.
- Run:

  ```bash
  python docs/examples/optical_cavity_feedback.py
  ```

## CLI replication

Reproduce the optical cavity workflow with the TNFR CLI using the canonical tokens stored in
[`sequence.json`](sequence.json) and the configuration file [`config.json`](config.json).

```bash
tnfr sequence \
  --nodes 3 --topology ring --seed 1 \
  --sequence-file docs/examples/sequence.json \
  --config docs/examples/config.json \
  --save-history history.json
```

Token legend:

| Token  | English operator |
| ------ | ---------------- |
| `AL`   | Emission         |
| `EN`   | Reception        |
| `IL`   | Coherence        |
| `UM`   | Coupling         |
| `RA`   | Resonance        |
| `SHA`  | Silence          |
| `NAV`  | Transition       |
| `OZ`   | Dissonance       |
| `ZHIR` | Mutation         |

The CLI run writes telemetry to `history.json`, mirroring the metrics produced by the Python
scripts. Inspect `W_stats` and `nodal_diag` entries to correlate coherence spans with node
states.
