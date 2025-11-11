How to reproduce TNFR results
=============================

The canonical TNFR experiments are reproducible when the environment, operator
sequences, and telemetry capture follow the conventions described here. This
page summarises the exact shell commands and Python snippets we run in CI to
obtain deterministic coherence measurements, ΔNFR traces, and Sense index (Si)
series.

.. contents::
   :local:
   :depth: 2

Environment preparation
-----------------------

TNFR requires Python 3.9 or newer. The commands below create an isolated
virtual environment, install the editable package together with the optional
extras used by the documentation and doctest build, and warm the optional
modules (`numpy`, `orjson`) that unlock vectorised paths and JSON telemetry
exports.

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate
   (.venv) $ pip install --upgrade pip
   (.venv) $ pip install -e '.[docs,numpy,test]'
   (.venv) $ pip install orjson  # optional but silences telemetry warnings

When reproducing historical experiments, pin the TNFR version and dependency
set using the published lock files or the `pyproject.toml` commit recorded in
`TNFR.pdf`. The editable install keeps the local sources and the interpreter in
sync, so rerunning doctests or Sphinx builds reflects the latest changes.

Running a scripted experiment
-----------------------------

The snippet below initialises a single coherent node, applies the canonical
operator sequence ``Emission → Reception → Coherence → Resonance → Silence`` and
stores the resulting coherence, ΔNFR and Si measurements in
``artifacts/tnfr_metrics.json``. The helper temporarily exposes the EPI value as
an explicit scalar so doctests can manipulate it without writing the full
BEPI tensor. Seeds are stored in the graph metadata and reused across runs.

.. code-block:: console

   (.venv) $ mkdir -p artifacts
   (.venv) $ python - <<'PY'
   import json
   import logging
   from contextlib import contextmanager

   from tnfr.alias import get_attr, set_attr
   from tnfr.constants import (
       DNFR_PRIMARY,
       EPI_PRIMARY,
       THETA_PRIMARY,
       VF_PRIMARY,
   )
   from tnfr.constants.aliases import ALIAS_EPI
   from tnfr.metrics.common import compute_coherence
   from tnfr.metrics.sense_index import compute_Si
   from tnfr.node import AttrSpec, NodeNX
   from tnfr.structural import (
       Coherence,
       Emission,
       Reception,
       Resonance,
       Silence,
       create_nfr,
       run_sequence,
   )

   logging.getLogger("tnfr.utils.init").setLevel(logging.ERROR)

   @contextmanager
   def scalar_epi_descriptor():
       original = NodeNX.EPI
       scalar_spec = AttrSpec(
           aliases=ALIAS_EPI,
           default=0.0,
           getter=get_attr,
           setter=set_attr,
           to_python=float,
           to_storage=float,
       )
       NodeNX.EPI = scalar_spec.build_property()
       try:
           yield
       finally:
           NodeNX.EPI = original

   with scalar_epi_descriptor():
       graph, node = create_nfr("seed", epi=0.35, vf=1.5, theta=0.0)
       graph.graph["RANDOM_SEED"] = 7
       operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
       run_sequence(graph, node, operators)
       coherence, mean_dnfr, mean_depi = compute_coherence(graph, return_means=True)
       si_values = dict(zip(graph.nodes, compute_Si(graph)))
       payload = {
           "node": node,
           "seed": graph.graph["RANDOM_SEED"],
           "operators": [op.name for op in operators],
           "EPI": graph.nodes[node][EPI_PRIMARY],
           "nu_f": graph.nodes[node][VF_PRIMARY],
           "theta": graph.nodes[node][THETA_PRIMARY],
           "delta_nfr": graph.nodes[node][DNFR_PRIMARY],
           "coherence": coherence,
           "delta_nfr_mean": mean_dnfr,
           "dEPI_dt_mean": mean_depi,
           "sense_index": float(si_values[node]),
       }
   with open("artifacts/tnfr_metrics.json", "w", encoding="utf-8") as fh:
       json.dump(payload, fh, indent=2, sort_keys=True)
   PY

The resulting JSON file lists every scalar required to trace the nodal equation
``∂EPI/∂t = νf · ΔNFR(t)``. For multi-node experiments, repeat the
``create_nfr``/``run_sequence`` block per node, append edges through
``tnfr.node.add_edge``, and persist the per-node dictionaries.

Capturing telemetry from the CLI
--------------------------------

The CLI reproduces the same experiments when it is launched from an environment
that already imported the scalar EPI descriptor (the helper above can be placed
in a bootstrap module and executed via ``python -m`` before invoking ``tnfr``).
A minimal run that records the glyph history, ΔNFR trace, and Si evolution looks
like this:

.. code-block:: console

   (.venv) $ python - <<'PY'
   """Patch NodeNX to expose scalar EPI before spawning the CLI."""
   from tnfr.alias import get_attr, set_attr
   from tnfr.constants.aliases import ALIAS_EPI
   from tnfr.node import AttrSpec, NodeNX

   NodeNX.EPI = AttrSpec(
       aliases=ALIAS_EPI,
       default=0.0,
       getter=get_attr,
       setter=set_attr,
       to_python=float,
       to_storage=float,
   ).build_property()
   PY
   (.venv) $ cat <<'JSON' > sequence.json
   [
     "emission",
     "reception",
     "coherence",
     "resonance",
     "silence"
   ]
   JSON
   (.venv) $ PYTHONPATH=src python -m tnfr.cli run \
       --nodes 24 \
       --topology ring \
       --seed 11 \
       --steps 180 \
       --sequence-file sequence.json \
       --save-history artifacts/history.json \
       --summary

``artifacts/history.json`` contains the per-step telemetry, including ``C(t)``,
ΔNFR magnitudes, phase angles, and Si. The ``--summary`` switch prints a compact
overview with the global coherence ``Tg``, average latency, and the glyphogram
series trimmed to the default limit of 128 samples. Repeat the command with
modified seeds or operator presets to profile different scenarios.

Verifying reproducibility
-------------------------

Re-run the scripted experiment (or the CLI variant) and confirm that the
telemetry is invariant under identical seeds by comparing the resulting files.
A simple Python helper checks for byte-for-byte equality and highlights any
fields that diverge beyond floating-point rounding.

.. code-block:: console

   (.venv) $ cp artifacts/tnfr_metrics.json artifacts/tnfr_metrics.baseline.json
   (.venv) $ python experiment.py  # rerun with the same configuration
   (.venv) $ python - <<'PY'
   import json
   from pathlib import Path

   baseline = json.loads(Path("artifacts/tnfr_metrics.baseline.json").read_text())
   current = json.loads(Path("artifacts/tnfr_metrics.json").read_text())
   assert baseline == current, "telemetry drift detected"
   PY

When multiple experiments are involved, promote the comparison to a table that
logs the seed, topology, and ΔNFR/Si deltas across runs. Commit these artifacts
alongside the manuscript or release to let reviewers and the CI pipeline retrace
the exact coherence profile.

Doctest: end-to-end API run
---------------------------

 The doctest below mirrors the scripted experiment so that ``sphinx-build -b
 doctest`` validates the workflow during documentation builds.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import logging
   >>> from contextlib import contextmanager
   >>> from tnfr.alias import get_attr, set_attr
   >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
   >>> from tnfr.constants.aliases import ALIAS_EPI
   >>> from tnfr.metrics.common import compute_coherence
   >>> from tnfr.metrics.sense_index import compute_Si
   >>> from tnfr.node import AttrSpec, NodeNX
   >>> from tnfr.structural import (
   ...     Coherence,
   ...     Emission,
   ...     Reception,
   ...     Resonance,
   ...     Silence,
   ...     create_nfr,
   ...     run_sequence,
   ... )
   >>> logging.getLogger("tnfr.utils.init").setLevel(logging.ERROR)
   >>> @contextmanager
   ... def scalar_epi_descriptor():
   ...     original = NodeNX.EPI
   ...     scalar_spec = AttrSpec(
   ...         aliases=ALIAS_EPI,
   ...         default=0.0,
   ...         getter=get_attr,
   ...         setter=set_attr,
   ...         to_python=float,
   ...         to_storage=float,
   ...     )
   ...     NodeNX.EPI = scalar_spec.build_property()
   ...     try:
   ...         yield
   ...     finally:
   ...         NodeNX.EPI = original
   >>> with scalar_epi_descriptor():
   ...     graph, node = create_nfr("doctest", epi=0.35, vf=1.5, theta=0.0)
   ...     ops = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
   ...     run_sequence(graph, node, ops)
   ...     coherence, mean_dnfr, mean_depi = compute_coherence(graph, return_means=True)
   ...     si_values = dict(zip(graph.nodes, compute_Si(graph)))
   ...     round(graph.nodes[node][EPI_PRIMARY], 6), round(graph.nodes[node][VF_PRIMARY], 6), round(graph.nodes[node][THETA_PRIMARY], 6), round(graph.nodes[node][DNFR_PRIMARY], 6), round(coherence, 6), round(mean_dnfr, 6), round(mean_depi, 6), round(float(si_values[node]), 6)
   (0.4, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
