# Release notes

## 16.0.0 (glyph load Spanish history key removed)

- **Breaking change**: Removed the deprecated ``"glyph_load_estab"`` history
  key. Metrics initialisation and the coherence observers now raise
  :class:`ValueError` as soon as the legacy identifier appears in a payload,
  preventing silent mirroring into ``"glyph_load_stabilizers"``.
- Migration guidance: audit stored histories and rename the key **before**
  loading a graph into this release. A minimal preprocessing snippet::

      history = payload.get("history", {})
      legacy = history.pop("glyph_load_estab", None)
      if isinstance(legacy, list) and "glyph_load_stabilizers" not in history:
          history["glyph_load_stabilizers"] = legacy

  Persist the rewritten payloads so downstream tooling only reads the English
  identifier. The project does not ship an automated helper for this upgrade;
  integrators should run the snippet (or an equivalent data migration) on any
  archived graphs prior to installing version 16.0.0.

## 14.0.0 (Spanish compatibility messaging retired)

- Finalised the English-only surface by removing Spanish-specific guidance from
  :mod:`tnfr.alias`, :mod:`tnfr.metrics.sense_index`, and the operator registry
  modules. Alias helpers now ignore untranslated payloads instead of raising
  bespoke errors and the sense index validates sensitivity mappings using
  generic key checks.
- Dropped the compatibility accessors in
  :mod:`tnfr.config.constants`, :mod:`tnfr.config.operator_names`, and
  :mod:`tnfr.operators.registry`. Accessing retired identifiers now surfaces the
  standard :class:`AttributeError` without custom wording.
- Documented the retirement timeline for
  :mod:`tnfr.utils.migrations`, which remains available for archival upgrades
  until ``tnfr`` 15.0.0 completes the migration window.
- Updated guides and release notes to describe the final English-only contract
  and the requirement to normalise archives with the compatibility helpers.

## 13.1.0 (preset legacy tuple removed)

- **Breaking change**: Removed the exported
  :data:`tnfr.config.presets.REMOVED_PRESET_NAMES` tuple now that Spanish preset
  identifiers are no longer recognised. Downstream tooling that introspected the
  tuple for migration support should ship its own static mapping.
- The :func:`tnfr.config.presets.get_preset` helper only consults canonical
  English identifiers. Legacy Spanish inputs (for example,
  ``get_preset('arranque_resonante')``) now raise ``KeyError('Preset not found: …')``
  without additional guidance, matching the behaviour for any unknown preset.

## 13.0.0 (selector norms alias removed)

- **Breaking change**: Removed the deprecated
  :func:`tnfr.selector._norms_para_selector` alias. Callers must import and use
  :func:`tnfr.selector._selector_norms` directly to fetch ΔNFR and acceleration
  maxima.
- Updated selector utilities documentation and tests to reference only the
  English helper so downstream projects surface the rename during upgrades.

## 12.1.0 (selector norms helper renamed)

- Renamed the selector norms helper to the English-only
  :func:`tnfr.selector._selector_norms` identifier to align selector internals
  with the ongoing terminology migration.
- Added a temporary compatibility shim for the legacy Spanish helper name that
  emitted :class:`DeprecationWarning` ahead of its removal in version 13.0.0.
- Updated :mod:`tnfr.dynamics` and the selector unit tests to consume the new
  helper, keeping the cached norms behaviour unchanged.

## 12.0.0 (diagnosis state Spanish shim removed)

- Removed the :func:`tnfr.constants.enable_spanish_state_tokens` and
  :func:`tnfr.constants.disable_spanish_state_tokens` compatibility helpers along
  with the ``TNFR_ENABLE_SPANISH_STATE_TOKENS`` environment flag. The diagnosis
  pipeline now rejects legacy Spanish literals instead of silently rewriting
  them at runtime.
- :func:`tnfr.constants.normalise_state_token` accepts only the canonical English
  tokens (``"stable"``, ``"transition"``, ``"dissonant"``) and raises
  :class:`ValueError` when historical payloads still carry Spanish values. The
  stricter contract propagates to
  :mod:`tnfr.metrics.diagnosis`, :mod:`tnfr.dynamics`, and
  :mod:`tnfr.glyph_history`, allowing integrations to surface explicit
  migrations instead of implicit rewrites.
- **Breaking change migration guidance**: run a preprocessing pass that rewrites
  stored states before upgrading, for example::

      SPANISH_STATE_TOKEN_MAP = {
          "est" "able": "stable",
          "trans" "icion": "transition",
          "transici" "\u00f3n": "transition",
          "diso" "nante": "dissonant",
      }

      def upgrade_state_token(value: str) -> str:
          token = value.strip().lower()
          if token in SPANISH_STATE_TOKEN_MAP:
              return SPANISH_STATE_TOKEN_MAP[token]
          if token in {"stable", "transition", "dissonant"}:
              return token
          raise ValueError(f"Unsupported diagnosis state: {value!r}")

      payload["state"] = upgrade_state_token(payload["state"])

  Persist the rewritten payloads before installing **TNFR 12.0.0** to avoid the
  new ``ValueError`` exceptions when loading historical archives.

## 11.2.0 (operator collections English-only)

- Removed the Spanish compatibility aliases from
  :mod:`tnfr.config.operator_names`. Accessing the retired names now raises
  :class:`AttributeError` pointing to the canonical English constant.
- Dropped the ``OPERAD<span></span>ORES`` alias from :mod:`tnfr.operators.registry`; only the
  English :data:`OPERATORS` registry is exported.
- Updated tests and helpers to enforce the English-only contract for operator
  collections, reflecting the final step in the migration announced in earlier
  releases.

## 11.1.0 (glyph load Spanish aggregates removed)

- :func:`tnfr.observers.glyph_load` now reports only the English aggregate
  keys ``"_stabilizers"`` and ``"_disruptors"``. The Spanish compatibility
  aliases were removed along with the runtime mirroring logic.
- Consumers in :mod:`tnfr.metrics.coherence` and :mod:`tnfr.dynamics` now read
  the English keys exclusively. Custom integrations should update any
  post-processing code that still expected ``"_estabilizadores"`` or
  ``"_disruptivos"`` entries.
- Updated the structural and metrics unit tests to enforce the English-only
  contract and removed the fixtures that patched Spanish aggregate labels.

## 11.0.0 (Si dispersion legacy keys removed)

- Removed the Spanish ``dSi_ddisp_fase`` attribute from the sense index
  sensitivity cache. Loading graphs or configuration payloads that still
  define the legacy key now raises :class:`ValueError` with guidance to use the
  English ``dSi_dphase_disp`` identifier.
- Updated :func:`tnfr.metrics.sense_index.compute_Si_node` so the Spanish
  ``disp_fase`` keyword argument is rejected with :class:`TypeError`. Callers
  must provide the ``phase_dispersion`` keyword when invoking the helper.
- Added migration guidance to the README for rewriting stored Si sensitivity
  mappings and configuration files that still carry the legacy identifiers.
- Migration snippet for historical graphs::

      sensitivity = G.graph.get("_Si_sensitivity") or {}
      if "dSi_ddisp_fase" in sensitivity:
          value = sensitivity.pop("dSi_ddisp_fase")
          sensitivity.setdefault("dSi_dphase_disp", value)

      # Persist the updated graph payload before upgrading the engine

## 10.0.0 (remesh stability window keyword removal)

- Removed the Spanish ``"pasos_" "est" "ables_consecutivos"`` keyword from
  :func:`tnfr.operators.apply_remesh_if_globally_stable`. Passing the legacy
  identifier now raises :class:`TypeError` with guidance to use the English
  ``stable_step_window`` parameter.
- Updated :mod:`tnfr.operators` documentation, telemetry guidance, and
  structural tests to reference only ``stable_step_window``.
- Published a migration guide covering the required code updates and how to
  audit stored configurations. See :doc:`getting-started/migrating-remesh-window`
  for detailed steps.

## 15.0.0 (legacy migration helpers removed)

- Finalised the English-only payload contract by removing
  :func:`tnfr.utils.migrations.migrate_legacy_phase_attributes` and
  :func:`tnfr.utils.migrations.migrate_legacy_remesh_cooldown`. Projects must now
  persist ``"theta"``, ``"phase"`` and ``"REMESH_COOLDOWN_WINDOW"`` directly
  because the helpers no longer rewrite ``"fase"``, ``"θ"`` or
  ``"REMESH_COOLDOWN_VENTANA"``.
- The archival migration window announced in TNFR 14.x expired on 2025-03-31.
  Upgrade pipelines should refuse to import graphs that still contain the
  Spanish keys instead of attempting a best-effort rewrite.
- Recommended pre-upgrade step::

      def migrate_payload(node_data: dict[str, float]) -> dict[str, float]:
          if "fase" in node_data:
              node_data["theta"] = float(node_data.pop("fase"))
          if "θ" in node_data:
              node_data["theta"] = float(node_data.pop("θ"))
          node_data["phase"] = float(node_data.get("theta", 0.0))
          return node_data

      def migrate_graph(G):
          if "REMESH_COOLDOWN_VENTANA" in G.graph:
              G.graph["REMESH_COOLDOWN_WINDOW"] = int(
                  G.graph.pop("REMESH_COOLDOWN_VENTANA")
              )
          for _, data in G.nodes(data=True):
              migrate_payload(data)

- Added documentation in :doc:`getting-started/migrating-remesh-window` that
  summarises the deadline and required checks before adopting this release.

## 9.0.0 (canonical preset rename)

- Renamed the canonical tutorial preset to the English-only identifier
  ``"canonical_example"``. The Spanish ``"ejemplo_canonico"`` alias now raises a
  :class:`KeyError` pointing to the supported name instead of being silently
  resolved.
- Updated :mod:`tnfr.execution` so :data:`tnfr.execution.CANONICAL_PRESET_NAME`
  exposes the English identifier, aligning the helper with
  :mod:`tnfr.config.presets`.
- Simplified the preset resolution layer by removing the remaining runtime
  aliases. ``get_preset()`` now rejects the retired identifiers with explicit
  guidance and the CLI surfaces the same migration message.
- Revised the CLI help strings, error handling, and documentation to mention
  only the English preset names. Downstream automation should update any stored
  configuration that still references ``"ejemplo_canonico"``.

## 8.1.0 (remesh cooldown alias removal)

- Removed the Spanish ``"REMESH_COOLDOWN_VENTANA"`` alias from
  :mod:`tnfr.constants.core.RemeshDefaults` and from the remesh operator
  pipeline. Configuration loaders and runtime helpers now require the English
  ``"REMESH_COOLDOWN_WINDOW"`` key and raise :class:`ValueError` when the
  legacy attribute is encountered, pointing to the migration utility below.
- Added :func:`tnfr.utils.migrate_legacy_remesh_cooldown` to rewrite persisted
  graphs in place. The helper removes the legacy key and promotes its value to
  the English attribute when necessary so stored payloads can be upgraded
  before running the new release.
- Updated tests and documentation to reflect the English-only remesh cooldown
  contract.

  Migration snippet::

      from tnfr.utils import migrate_legacy_remesh_cooldown

      G = load_graph()  # application-specific loader
      migrate_legacy_remesh_cooldown(G)
      inject_defaults(G)  # optional, keeps defaults in sync

## 8.0.0 (phase alias enforcement)

- Finalised the phase attribute migration by rejecting the Spanish ``"fase"``
  node key. Access helpers in :mod:`tnfr.alias` now operate strictly on the
  English ``"theta"``/``"phase"`` attributes and raise
  :class:`ValueError` when the legacy key is encountered.
- Added :func:`tnfr.utils.migrate_legacy_phase_attributes` to help upgrade
  stored graphs. The helper rewrites ``"fase"`` and ``"θ"`` payloads in place,
  populating the canonical English keys before interacting with the alias
  layer.
- Updated documentation and tests to reflect the English-only phase contract
  and removed the automatic ``"fase"`` rewrite shims.

## 7.0.1 (English deprecation messaging)

- Reworded the remaining deprecation warnings and validation errors that still
  surfaced Spanish text. Deprecation shims in :mod:`tnfr.constants_glyphs` and
  :mod:`tnfr.presets` now emit English guidance, and the operator registry plus
  metrics export helpers raise English-only :class:`ValueError` messages for
  unsupported usage.

## 7.0.0 (Spanish identifiers removed)

- Removed the Spanish glyph constants ``ESTABILIZADORES`` and ``DISRUPTIVOS``
  from :mod:`tnfr.config.constants`. Import the English
  :data:`tnfr.config.constants.STABILIZERS` and
  :data:`tnfr.config.constants.DISRUPTORS` names instead. Accessing the old
  identifiers now raises :class:`AttributeError` after emitting a final
  :class:`FutureWarning` explaining the required substitution.
- Finalised the state token migration. Spanish literals now require an explicit
  opt-in via :func:`tnfr.constants.enable_spanish_state_tokens` or by setting
  the :envvar:`TNFR_ENABLE_SPANISH_STATE_TOKENS` environment variable. The shim
  warns with :class:`FutureWarning` and is scheduled for removal in TNFR 8.0.
- Removed the ``SPANISH_PRESET_ALIASES`` helper and the runtime resolution of
  Spanish preset identifiers. Calls such as ``get_preset('arranque_resonante')``
  now raise :class:`KeyError` indicating the English replacement. Only English
  preset names remain in the public API.
- Updated tests and documentation to reflect the English-only contract across
  glyph constants, preset helpers, and diagnostic state utilities.

## 6.1.0 (preset alias deprecation window)

- Announced the removal of the Spanish preset identifiers
  (``arranque_resonante``, ``mutacion_contenida``, ``exploracion_acople``) in
  **TNFR 7.0**. The engine now emits :class:`FutureWarning` when the legacy
  names are resolved so pipelines can surface the upcoming breakage.
- Added the ``tnfr.config.presets.SPANISH_PRESET_ALIASES`` mapping to help
  audit configurations. Existing presets should switch to the English
  equivalents (``resonant_bootstrap``, ``contained_mutation``,
  ``coupling_exploration``) before upgrading to 7.0. The helper was removed in
  TNFR 7.0 once the migration period ended; downstream projects should now keep
  a local mapping during the final substitution pass.
- Migration helper: update YAML/JSON payloads or CLI arguments with a simple
  substitution pass. The following snippet illustrates how to migrate data once
  the mapping is defined locally::

      SPANISH_PRESET_ALIASES = {
          "arranque_resonante": "resonant_bootstrap",
          "mutacion_contenida": "contained_mutation",
          "exploracion_acople": "coupling_exploration",
      }

      def normalize_preset_name(name: str) -> str:
          return SPANISH_PRESET_ALIASES.get(name, name)

      user_config["preset"] = normalize_preset_name(user_config["preset"])

## 6.0.0 (Nodo aliases removed)

- Removed the Spanish module-level aliases ``tnfr.node.NodoNX`` and
  ``tnfr.node.NodoProtocol``. Code importing those symbols must switch to the
  canonical :class:`tnfr.node.NodeNX` and :class:`tnfr.node.NodeProtocol`
  definitions immediately.
- Deleted :func:`tnfr.utils.get_nodonx`. Downstream helpers should import and
  call :func:`tnfr.utils.get_nodenx`, which keeps returning
  :class:`tnfr.node.NodeNX` through the cached import layer.
- Pruned the typing stubs, tests, and utilities that referenced the Spanish
  names so static analysis and runtime behaviour now agree on the English-only
  surface area.
- Published the backward-incompatible change as **TNFR 6.0.0** to honour the
  semantic versioning contract and flag the immediate API removal.

## 5.0.0 (prepare_network alias retired)

- Removed the Spanish helper alias ``tnfr.preparar_red``. The network
  preparation pipeline now ships exclusively under the English
  :func:`tnfr.prepare_network` name. Codebases that still relied on the
  alias must update their imports before upgrading.
- Updated the typing stubs, integration tests, and documentation to
  reflect the canonical helper set.
- Bumped the package version to **5.0.0** to flag the
  backward-incompatible API change.

## 2.0.0 (Spanish alias removal)

- Removed the Spanish compatibility tables from :mod:`tnfr.config.operator_names`.
  Only the English tokens (``emission``, ``reception``, ``coherence``, etc.) are
  accepted by validation helpers, the operator registry, and the CLI parser.
- Removed :mod:`tnfr.operators.compat` and the Spanish class wrappers that
  previously emitted :class:`DeprecationWarning`. All structural orchestration
  must now import the English operator classes from
  :mod:`tnfr.operators.definitions` or :mod:`tnfr.structural`.
- Trimmed Spanish re-exports from :mod:`tnfr.structural` and its typing stubs so
  only the English operator classes remain in the public ``__all__``.
- Impacted entry points:
  * Stored operator sequences (YAML/JSON fixtures, CLI configs) must be rewritten
    to use the English identifiers.
  * Programmatic calls to :func:`tnfr.structural.run_sequence`,
    :func:`tnfr.validation.syntax.validate_sequence`, and
    :func:`tnfr.operators.registry.get_operator_class` will now reject Spanish
    tokens.
  * Import sites that referenced ``tnfr.operators.compat`` or the Spanish class
    names exported from :mod:`tnfr.structural` must update their imports to the
    English equivalents.
  * Diagnostics relying on ``TRANS<span></span>ICION`` should switch to the English
    ``TRANSITION`` constant from :mod:`tnfr.config.operator_names`.
- Versioning and communication plan:
  * Publish this change as **TNFR 2.0.0** and note the breaking removal in the
    release announcement.
  * No transition shims remain—migrating to the English tokens is mandatory
    before adopting 2.0.
  * Ship an upgrade checklist highlighting the required token substitutions and
    the removal of ``tnfr.operators.compat``.
  * Update API docs, tutorials, and CLI references to show only English tokens.

## 1.x compatibility window (historical)

- English operator identifiers became canonical in 1.x. The registry published
  the English tokens and the CLI expected the same literals while the Spanish
  class names remained available in :mod:`tnfr.operators.compat` as wrappers
  that raised :class:`DeprecationWarning`.

- Renamed the network preparation helper to `prepare_network` for
  consistency with the English-facing API. The previous Spanish name
  `preparar_red` emitted a :class:`DeprecationWarning` and has now been
  removed in **TNFR 5.0**. Use the English helper directly to stay
  within the supported contract.

- Unified the node wrappers under the English identifiers
  :class:`tnfr.node.NodeNX` and :class:`tnfr.node.NodeProtocol`. The Spanish
  counterparts (`NodoNX`, `NodoProtocol`) were deprecated in the 1.x cycle and
  have now been removed in **TNFR 6.0**. Import the English names directly to
  remain within the supported contract.

All other helpers continue to honour the existing dependency manifest
and import semantics.
