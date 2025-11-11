- Removed the deprecated helper compatibility layer; import helper utilities from `tnfr.utils`.
- Importing `tnfr.cache` or `tnfr.io` now raises ``ImportError``. Update imports to
  :mod:`tnfr.utils.cache` and :mod:`tnfr.utils.io` to keep relying on the supported API.
