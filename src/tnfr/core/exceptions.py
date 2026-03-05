"""
TNFR Unified Exception Hierarchy.

Centralizes all custom exceptions to ensure consistent error handling
across the engine. All custom exceptions should inherit from TNFRError.
"""


class TNFRError(Exception):
    """Base class for all TNFR exceptions."""
    pass


class StructuralError(TNFRError):
    """Base class for structural/physics errors."""
    pass


class CoherenceError(StructuralError):
    """Raised when coherence constraints (U2, U6) are violated."""
    pass


class PhaseError(StructuralError):
    """Raised when phase coupling constraints (U3) are violated."""
    pass


class GrammarError(StructuralError):
    """Raised when operator sequences violate unified grammar (U1-U6)."""
    pass


class BackendError(TNFRError):
    """Base class for computational backend errors."""
    pass


class BackendUnavailableError(BackendError):
    """Raised when a requested backend cannot be initialized."""
    pass


class ConfigurationError(TNFRError):
    """Raised when configuration is invalid."""
    pass


class OptimizationError(TNFRError):
    """Raised when self-optimization fails."""
    pass

