"""Community tools for TNFR extension development and validation.

This package provides tools for community contributors to develop, validate,
and submit TNFR extensions.

Examples
--------
>>> from tools.community.extension_validator import ExtensionValidator
>>> from tnfr.extensions.medical import MedicalExtension
>>> 
>>> validator = ExtensionValidator()
>>> report = validator.validate_extension(MedicalExtension())
>>> print(f"Overall quality: {report.overall_score:.2f}")
"""

from __future__ import annotations

__all__ = []
