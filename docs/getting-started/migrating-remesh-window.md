# Migrating remesh stability window usage

TNFR 10.0.0 removes the transitional Spanish keyword
``pasos_estables_consecutivos`` from
:func:`tnfr.operators.apply_remesh_if_globally_stable`. The operator now
accepts only the English ``stable_step_window`` parameter. Calls that still use
or forward the Spanish keyword raise :class:`TypeError` immediately so the
deprecated configuration cannot silently slip through pipelines.

Legacy graphs that still expose the Spanish cooldown metadata **had to be
migrated before 2025-03-31**. That date marked the end of the archival
compatibility window communicated in TNFR 14.x. Starting with ``tnfr`` 15.0.0
the runtime no longer ships :func:`tnfr.utils.migrations.migrate_legacy_remesh_cooldown`
or the phase attribute shim, so persisted payloads must already use the English
keys.

## Who is affected?

- Applications that invoked
  ``apply_remesh_if_globally_stable(G, pasos_estables_consecutivos=...)``.
- Configuration loaders that surfaced the Spanish name as part of dynamic
  keyword expansion.
- Stored automation artifacts (YAML/JSON, notebooks) that preserved the legacy
  keyword for reproducibility.

## Migration steps

1. Replace every occurrence of the Spanish keyword with the canonical English
   parameter. The semantic contract is unchanged::

       # Before (TNFR <= 9.x)
       apply_remesh_if_globally_stable(G, pasos_estables_consecutivos=5)

       # After (TNFR >= 10.0)
       apply_remesh_if_globally_stable(G, stable_step_window=5)

2. If you rely on user-provided dictionaries that may contain the legacy
   identifier, normalise the payload before calling the operator::

       def normalize_remesh_kwargs(kwargs: dict) -> dict:
           if "pasos_estables_consecutivos" in kwargs:
               kwargs = dict(kwargs)  # shallow copy if shared
               kwargs["stable_step_window"] = kwargs.pop("pasos_estables_consecutivos")
           return kwargs

       apply_remesh_if_globally_stable(G, **normalize_remesh_kwargs(user_kwargs))

3. Audit persisted graphs or configs that reference the Spanish names **before
   upgrading to ``tnfr`` 15.0.0**. Without the helper, the update must happen in
   the stored artifact (for example by running a one-off script on the graph
   metadata) prior to importing the new release.

4. Verify that serialized graphs expose only ``"theta"``, ``"phase"`` and
   ``"REMESH_COOLDOWN_WINDOW"``. Any remaining ``"fase"``, ``"θ"`` or
   ``"REMESH_COOLDOWN_VENTANA"`` entries indicate the artifact still targets an
   unsupported contract.

## Verification checklist

- Unit or integration tests that previously asserted a deprecation warning
  should now expect :class:`TypeError` when the Spanish keyword is passed.
- Documentation, CLI help, and user-facing guidance must reference only the
  ``stable_step_window`` parameter.
- Downstream logs or telemetry that templated the Spanish name should be
  updated to keep observability messages aligned with the supported API.
- Graph validation pipelines must fail fast if a payload still contains the
  deprecated Spanish keys after the deadline, because the helpers are no longer
  available to rewrite them.

Following these steps ensures remesh orchestration remains stable while the
engine enforces the English-only parameter surface.
