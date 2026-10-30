"""Source-checkout discovery for optional applications outside the engine."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path


def bootstrap_application(package: str, directory: str) -> None:
    """Expose a checkout application only when no installed package exists.

    Importing the engine does not activate these applications. Their public
    adapters call this helper on demand; wheel installations require the
    optional package itself and do not contain the application source trees.
    """
    if package in sys.modules or find_spec(package) is not None:
        return
    engine_directory = Path(__file__).resolve().parent
    repository = engine_directory.parents[1]
    if repository / "src" / "tnfr" != engine_directory:
        return
    application = repository / "applications" / directory
    if (application / package / "__init__.py").is_file():
        location = str(application)
        if location not in sys.path:
            sys.path.insert(0, location)
