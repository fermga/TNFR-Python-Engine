# Release notes

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
  :class:`tnfr.node.NodeNX` and :class:`tnfr.node.NodeProtocol`. Their
  Spanish counterparts (`NodoNX`, `NodoProtocol`) remain available as
  compatibility aliases that raise :class:`DeprecationWarning` and will
  be removed on **2025-12-01**. Migrate imports and type annotations to
  the English names to avoid churn when the compatibility window closes.

All other helpers continue to honour the existing dependency manifest
and import semantics.
