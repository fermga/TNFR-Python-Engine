"""
TNFR Unified Exception Hierarchy.

Centralizes all custom exceptions to ensure consistent error handling
across the engine. All custom exceptions should inherit from TNFRError.
"""

class TNFRError(Exception):
    """Base class for all TNFR exceptions."""

class StructuralError(TNFRError):
    """Base class for structural/physics errors."""

class CoherenceError(StructuralError):
    """Raised when coherence constraints (U2, U6) are violated."""

class PhaseError(StructuralError):
    """Raised when phase coupling constraints (U3) are violated."""

class GrammarError(StructuralError):
    """Raised when operator sequences violate unified grammar (U1-U6)."""

class BackendError(TNFRError):
    """Base class for computational backend errors."""

class BackendUnavailableError(BackendError):
    """Raised when a requested backend cannot be initialized."""

class ConfigurationError(TNFRError):
    """Raised when configuration is invalid."""

class OptimizationError(TNFRError):
    """Raised when self-optimization fails."""

