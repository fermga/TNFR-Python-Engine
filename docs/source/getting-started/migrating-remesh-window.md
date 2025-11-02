# Migrating remesh stability window usage

TNFR 10.0.0 removes the transitional alias for the remesh stability window
parameter from :func:`tnfr.operators.apply_remesh_if_globally_stable`. The
operator now accepts only the canonical ``stable_step_window`` argument. Calls
that forward any other keyword raise :class:`TypeError` immediately so deprecated
configuration cannot slip through pipelines.

Legacy graphs that still expose non-canonical cooldown metadata **had to be
migrated before 2025-03-31**. That date marked the end of the archival
compatibility window communicated in TNFR 14.x. Starting with ``tnfr`` 15.0.0 the
runtime no longer ships the built-in remesh cooldown or phase attribute migration
helpers, so persisted payloads must already use the canonical keys when you
upgrade. Historical archives that missed the window need to be rewritten with an
external script before they are loaded into a modern runtime.

## Who is affected?

- Applications that invoked the operator with a deprecated alias, e.g.
  ``apply_remesh_if_globally_stable(G, **legacy_kwargs)`` where ``legacy_kwargs``
  forwards a non-English key.
- Configuration loaders that surfaced the legacy name as part of dynamic
  keyword expansion.
- Stored automation artifacts (YAML/JSON, notebooks) that preserved the legacy
  keyword for reproducibility.

## Migration steps

1. Replace every occurrence of the deprecated parameter name with the canonical
   ``stable_step_window`` argument. The semantic contract is unchanged::

       apply_remesh_if_globally_stable(G, stable_step_window=5)

2. If you rely on user-provided dictionaries that may contain non-canonical
   names, normalise the payload before calling the operator::

       def normalize_remesh_kwargs(kwargs: dict) -> dict:
           normalized = dict(kwargs)
           for key in tuple(normalized):
               if key != "stable_step_window" and key.lower().replace("-", "_") == "stable_step_window":
                   normalized["stable_step_window"] = normalized.pop(key)
           return normalized

       apply_remesh_if_globally_stable(G, **normalize_remesh_kwargs(user_kwargs))

3. Audit persisted graphs or configs that reference non-English names **before
   upgrading to ``tnfr`` 15.0.0**. Without the bundled helper, the update must
   happen in the stored artifact (for example by running a one-off script on the
   graph metadata) prior to importing the new release.

4. Verify that serialized graphs expose only ``"theta"``, ``"phase"`` and
   ``"REMESH_COOLDOWN_WINDOW"``. Any other synonym indicates the artifact still
   targets an unsupported contract.

### Rewriting archives after the helper removal

When the bundled migration helpers shipped in ``tnfr`` 14.x they performed an
in-place rewrite of legacy keys. With their removal you need to run an external
upgrade step before the graph interacts with the runtime. A lightweight script
that walks the stored payload and promotes the canonical keys is sufficient::

    import json

    def rewrite_remesh_metadata(payload: dict) -> dict:
        upgraded = dict(payload)
        if "θ" in upgraded and "theta" not in upgraded:
            upgraded["theta"] = upgraded.pop("θ")
        legacy_window = upgraded.pop("ventana_enfriamiento", None)
        if legacy_window is not None and "REMESH_COOLDOWN_WINDOW" not in upgraded:
            upgraded["REMESH_COOLDOWN_WINDOW"] = legacy_window
        return upgraded

    with open("graph.json") as fh:
        data = json.load(fh)
    data["graph"] = rewrite_remesh_metadata(data["graph"])
    with open("graph.json", "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)

Integrations can adapt the sample to their storage format (for example, NetworkX
pickles or database rows). The important part is that the normalization happens
*before* the graph loads into ``tnfr`` 15.0.0 or later.

## Verification checklist

- Unit or integration tests that previously asserted a deprecation warning
  should now expect :class:`TypeError` when the legacy keyword is passed.
- Documentation, CLI help, and user-facing guidance must reference only the
  ``stable_step_window`` parameter.
- Downstream logs or telemetry that templated the legacy name should be
  updated to keep observability messages aligned with the supported API.
- Graph validation pipelines must fail fast if a payload still contains the
  deprecated legacy keys after the deadline, because the automatic rewrites are
  no longer part of the runtime.

Following these steps ensures remesh orchestration remains stable while the
engine enforces the English-only parameter surface.
