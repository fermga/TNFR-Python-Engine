# Release notes

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
  * Diagnostics relying on ``TRANSICION`` should switch to the English
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
