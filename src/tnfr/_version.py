"""Runtime version discovery for :mod:`tnfr`."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Final

__all__ = ["__version__"]


def _read_version() -> str:
    """Resolve the published package version while preserving TNFR invariants."""

    env_version = os.environ.get("TNFR_VERSION")
    if env_version:
        return env_version

    try:
        return metadata.version("tnfr")
    except metadata.PackageNotFoundError:
        pass  # Fallback to alternative version sources

    try:  # pragma: no cover - only present in built distributions
        from . import _generated_version  # type: ignore
    except ImportError:  # pragma: no cover - optional artifact
        pass  # Generated version not available
    else:
        generated = getattr(_generated_version, "version", None)
        if isinstance(generated, str) and generated:
            return generated
        legacy = getattr(_generated_version, "__version__", None)
        if isinstance(legacy, str) and legacy:
            return legacy

    try:
        from setuptools_scm import get_version
    except Exception:  # pragma: no cover - optional dependency
        pass  # setuptools_scm not available
    else:
        try:
            return get_version(relative_to=__file__)
        except LookupError:
            pass  # No version found via setuptools_scm

    return "0.0.0"


__version__: Final[str] = _read_version()
