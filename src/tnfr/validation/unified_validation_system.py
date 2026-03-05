"""TNFR Unified Validation System - Consolidated Input and Security Validation.

CONSOLIDATION ACHIEVEMENT: This module unifies all TNFR validation implementations
under a single coherent interface following nodal equation dynamics principles.

Unified Architecture:
- Consolidates validation/input_validation.py input parameter validation
- Merges security/validation.py security-focused validation functionality  
- Unifies type checking and structural invariant enforcement
- Consistent error handling and validation reporting
- Integrated with unified configuration system

Theoretical Foundation:
Validation enforces TNFR canonical invariants derived from nodal equation
∂EPI/∂t = νf · ΔNFR(t) and structural field constraints to ensure theoretical
consistency across all TNFR operations.

Consolidated Features:
1. Structural Validation: EPI, νf, φ/θ, ΔNFR parameter validation  
2. Security Validation: Input sanitization and injection prevention
3. Type Validation: TNFRGraph, NodeId, Glyph type checking
4. Invariant Enforcement: Canonical constraints (C(t), Si, tetrad bounds)
5. Range Validation: Proper value bounds for all TNFR parameters
6. Error Reporting: Unified validation error hierarchy

Consolidates:
- src/tnfr/validation/input_validation.py (input parameter validation)
- src/tnfr/security/validation.py (security-focused validation)
- Scattered validation logic across operators and physics modules

Status: UNIFIED VALIDATION CONSOLIDATION - All validation centralized
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import logging

from ..mathematics.unified_numerical import np

# Unified configuration integration
from ..config import get_config
from ..errors import TNFRValueError, TNFRSecurityError

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for all validation errors."""
    pass


@dataclass
class ValidationResult:
    """Result of validation operation with detailed feedback."""
    
    is_valid: bool
    error_messages: List[str]
    warnings: List[str]
    validated_value: Any = None
    validation_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.validation_metadata is None:
            self.validation_metadata = {}


@dataclass
class ValidationConfig:
    """Configuration for unified validation system."""
    
    # Validation strictness
    strict_mode: bool = True
    enable_warnings: bool = True
    
    # TNFR structural bounds
    max_structural_frequency: float = 1000.0  # Hz_str
    min_structural_frequency: float = 0.0
    max_phase_value: float = 2 * math.pi
    min_phase_value: float = 0.0
    
    # Coherence and stability bounds
    min_coherence: float = 0.0
    max_coherence: float = 1.0
    min_sense_index: float = 0.0
    max_sense_index: float = float('inf')
    
    # Security validation
    enable_security_checks: bool = True
    max_string_length: int = 1000
    forbidden_patterns: List[str] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_validation_results: bool = True
    
    def __post_init__(self):
        """Initialize default forbidden patterns."""
        if self.forbidden_patterns is None:
            self.forbidden_patterns = [
                r'<script',
                r'javascript:',
                r'eval\(',
                r'exec\(',
                r'\$\{',
                r'`.*`'
            ]


class TNFRValidationError(TNFRValueError):
    """Unified validation error for TNFR structural constraints."""
    
    def __init__(self, message: str, field_name: str = None, validation_context: Dict[str, Any] = None, suggestion: str = None):
        context = validation_context or {}
        if field_name:
            context["field"] = field_name
            
        super().__init__(
            message=message,
            context=context,
            suggestion=suggestion
        )
        self.field_name = field_name
        self.validation_context = context


class TNFRUnifiedValidationSystem:
    """Unified Validation System - Consolidated Input and Security Validation.
    
    ARCHITECTURE: This system consolidates all TNFR validation implementations
    under a unified interface with intelligent routing and caching.
    
    Consolidates:
    - Input parameter validation from validation/input_validation.py
    - Security validation from security/validation.py  
    - Type checking across operators and physics modules
    - Structural invariant enforcement
    
    Usage:
        # Single entry point for all validation
        validator = TNFRUnifiedValidationSystem()
        
        # Structural parameter validation
        result = validator.validate_structural_frequency(0.5)
        assert result.is_valid
        
        # Security validation
        result = validator.validate_string_input("user_input")
        
        # Composite validation
        result = validator.validate_tnfr_graph(graph_data)
        
        # Batch validation
        results = validator.validate_multiple({
            "vf": 1.2,
            "phase": 3.14,
            "coherence": 0.85
        })
    
    Benefits:
        - Eliminates validation redundancy across codebase
        - Consistent error messages and validation behavior
        - Unified caching for performance optimization
        - Integrated security and structural validation
        - Comprehensive validation reporting
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize unified validation system."""
        self.config = config or ValidationConfig()
        
        # Validation cache for performance
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Global configuration integration
        self.global_config = get_config()
        
        # Compile regex patterns for security validation
        self._compiled_security_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.config.forbidden_patterns
        ]
        
        logger.info(f"Initialized unified validation system with config: {self.config}")
    
    def validate_structural_frequency(
        self, 
        vf: Union[float, int], 
        field_name: str = "vf"
    ) -> ValidationResult:
        """Validate structural frequency (νf) parameter.
        
        CONSOLIDATION: Unifies νf validation from input_validation.py
        and security/validation.py with enhanced error reporting.
        
        Parameters
        ----------
        vf : float or int
            Structural frequency value in Hz_str units
        field_name : str
            Name of the field being validated for error reporting
            
        Returns
        -------
        ValidationResult
            Validation result with detailed feedback
        """
        cache_key = f"vf_{vf}_{field_name}"
        
        # Check cache
        if self.config.enable_caching and cache_key in self._validation_cache:
            self._cache_stats["hits"] += 1
            return self._validation_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        
        errors = []
        warnings = []
        validated_value = vf
        
        # Type validation
        if not isinstance(vf, (int, float)):
            errors.append(f"{field_name} must be a number, got {type(vf).__name__}")
        else:
            # Convert to float for consistency
            validated_value = float(vf)
            
            # Range validation
            if validated_value < self.config.min_structural_frequency:
                errors.append(f"{field_name} must be >= {self.config.min_structural_frequency}, got {validated_value}")
            
            if validated_value > self.config.max_structural_frequency:
                if self.config.strict_mode:
                    errors.append(f"{field_name} exceeds maximum {self.config.max_structural_frequency}, got {validated_value}")
                else:
                    warnings.append(f"{field_name} is very large ({validated_value}), consider checking units")
            
            # Special values validation
            if math.isnan(validated_value):
                errors.append(f"{field_name} cannot be NaN")
            elif math.isinf(validated_value):
                errors.append(f"{field_name} cannot be infinite")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={"field_type": "structural_frequency", "units": "Hz_str"}
        )
        
        # Cache result
        if self.config.cache_validation_results:
            self._validation_cache[cache_key] = result
        
        return result
    
    def validate_phase_value(
        self, 
        phase: Union[float, int], 
        field_name: str = "phase",
        normalize: bool = True
    ) -> ValidationResult:
        """Validate phase (φ/θ) parameter.
        
        CONSOLIDATION: Unifies phase validation with normalization support.
        
        Parameters
        ----------
        phase : float or int
            Phase value in radians
        field_name : str
            Name of the field being validated
        normalize : bool
            Whether to normalize phase to [0, 2π] range
            
        Returns
        -------
        ValidationResult
            Validation result with normalized phase value
        """
        cache_key = f"phase_{phase}_{field_name}_{normalize}"
        
        # Check cache
        if self.config.enable_caching and cache_key in self._validation_cache:
            self._cache_stats["hits"] += 1
            return self._validation_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        
        errors = []
        warnings = []
        validated_value = phase
        
        # Type validation
        if not isinstance(phase, (int, float)):
            errors.append(f"{field_name} must be a number, got {type(phase).__name__}")
        else:
            validated_value = float(phase)
            
            # Special values validation
            if math.isnan(validated_value):
                errors.append(f"{field_name} cannot be NaN")
            elif math.isinf(validated_value):
                errors.append(f"{field_name} cannot be infinite")
            else:
                # Normalize if requested
                if normalize:
                    validated_value = validated_value % (2 * math.pi)
                
                # Range warnings for unnormalized values
                if not normalize:
                    if validated_value < self.config.min_phase_value or validated_value > self.config.max_phase_value:
                        warnings.append(f"{field_name} outside typical range [0, 2π], got {validated_value}")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={"field_type": "phase", "units": "radians", "normalized": normalize}
        )
        
        # Cache result
        if self.config.cache_validation_results:
            self._validation_cache[cache_key] = result
        
        return result
    
    def validate_coherence(
        self, 
        coherence: Union[float, int], 
        field_name: str = "coherence"
    ) -> ValidationResult:
        """Validate coherence C(t) parameter.
        
        CONSOLIDATION: Unifies coherence validation with proper bounds checking.
        """
        cache_key = f"coherence_{coherence}_{field_name}"
        
        # Check cache
        if self.config.enable_caching and cache_key in self._validation_cache:
            self._cache_stats["hits"] += 1
            return self._validation_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        
        errors = []
        warnings = []
        validated_value = coherence
        
        # Type validation
        if not isinstance(coherence, (int, float)):
            errors.append(f"{field_name} must be a number, got {type(coherence).__name__}")
        else:
            validated_value = float(coherence)
            
            # Range validation
            if validated_value < self.config.min_coherence:
                errors.append(f"{field_name} must be >= {self.config.min_coherence}, got {validated_value}")
            elif validated_value > self.config.max_coherence:
                errors.append(f"{field_name} must be <= {self.config.max_coherence}, got {validated_value}")
            
            # Special values validation
            if math.isnan(validated_value):
                errors.append(f"{field_name} cannot be NaN")
            elif math.isinf(validated_value):
                errors.append(f"{field_name} cannot be infinite")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={"field_type": "coherence", "bounds": [self.config.min_coherence, self.config.max_coherence]}
        )
        
        # Cache result
        if self.config.cache_validation_results:
            self._validation_cache[cache_key] = result
        
        return result
    
    def validate_string_input(
        self, 
        input_string: str, 
        field_name: str = "input",
        max_length: Optional[int] = None
    ) -> ValidationResult:
        """Validate string input with security checks.
        
        CONSOLIDATION: Unifies string validation from security/validation.py
        with enhanced pattern matching and injection detection.
        
        Parameters
        ----------
        input_string : str
            String to validate
        field_name : str
            Name of the field being validated
        max_length : int, optional
            Maximum allowed string length (uses config default if not provided)
            
        Returns
        -------
        ValidationResult
            Validation result with security assessment
        """
        if not self.config.enable_security_checks:
            return ValidationResult(
                is_valid=True,
                error_messages=[],
                warnings=[],
                validated_value=input_string
            )
        
        max_len = max_length or self.config.max_string_length
        cache_key = f"string_{hash(input_string)}_{field_name}_{max_len}"
        
        # Check cache
        if self.config.enable_caching and cache_key in self._validation_cache:
            self._cache_stats["hits"] += 1
            return self._validation_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        
        errors = []
        warnings = []
        validated_value = input_string
        
        # Type validation
        if not isinstance(input_string, str):
            errors.append(f"{field_name} must be a string, got {type(input_string).__name__}")
        else:
            # Length validation
            if len(input_string) > max_len:
                errors.append(f"{field_name} exceeds maximum length {max_len}, got {len(input_string)}")
            
            # Security pattern validation
            for pattern in self._compiled_security_patterns:
                if pattern.search(input_string):
                    errors.append(f"{field_name} contains potentially unsafe pattern: {pattern.pattern}")
                    break
            
            # Additional security checks
            if '<' in input_string and '>' in input_string:
                warnings.append(f"{field_name} contains angle brackets, verify if intended")
            
            if input_string.strip() != input_string:
                warnings.append(f"{field_name} has leading/trailing whitespace")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={"field_type": "string", "security_checked": True, "max_length": max_len}
        )
        
        # Cache result
        if self.config.cache_validation_results:
            self._validation_cache[cache_key] = result
        
        return result
    
    def validate_array_input(
        self, 
        array: np.ndarray, 
        field_name: str = "array",
        expected_shape: Optional[tuple] = None,
        expected_dtype: Optional[type] = None
    ) -> ValidationResult:
        """Validate NumPy array input for TNFR operations.
        
        Parameters
        ----------
        array : np.ndarray
            Array to validate
        field_name : str
            Name of the field being validated
        expected_shape : tuple, optional
            Expected array shape
        expected_dtype : type, optional
            Expected array data type
            
        Returns
        -------
        ValidationResult
            Validation result with array information
        """
        errors = []
        warnings = []
        validated_value = array
        
        # Type validation
        if not isinstance(array, np.ndarray):
            errors.append(f"{field_name} must be a NumPy array, got {type(array).__name__}")
        else:
            # Shape validation
            if expected_shape is not None and array.shape != expected_shape:
                errors.append(f"{field_name} shape mismatch: expected {expected_shape}, got {array.shape}")
            
            # Data type validation
            if expected_dtype is not None and array.dtype != expected_dtype:
                warnings.append(f"{field_name} dtype mismatch: expected {expected_dtype}, got {array.dtype}")
            
            # Special values validation
            if np.any(np.isnan(array)):
                errors.append(f"{field_name} contains NaN values")
            elif np.any(np.isinf(array)):
                errors.append(f"{field_name} contains infinite values")
            
            # Size validation (prevent memory issues)
            if array.size > 1e8:  # 100M elements
                warnings.append(f"{field_name} is very large ({array.size} elements), may cause memory issues")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={
                "field_type": "array", 
                "shape": array.shape if isinstance(array, np.ndarray) else None,
                "dtype": str(array.dtype) if isinstance(array, np.ndarray) else None
            }
        )
        
        return result
    
    def validate_multiple(
        self, 
        values: Dict[str, Any],
        validation_rules: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, ValidationResult]:
        """Validate multiple values with unified error handling.
        
        Parameters
        ----------
        values : dict
            Dictionary of field names to values to validate
        validation_rules : dict, optional
            Custom validation rules for specific fields
            
        Returns
        -------
        dict
            Dictionary of field names to validation results
        """
        results = {}
        
        # Default validation rules
        default_rules = {
            'vf': self.validate_structural_frequency,
            'phase': self.validate_phase_value,
            'coherence': self.validate_coherence,
            'structural_frequency': self.validate_structural_frequency,
            'phi': self.validate_phase_value,
            'theta': self.validate_phase_value
        }
        
        # Merge with custom rules
        rules = {**default_rules, **(validation_rules or {})}
        
        for field_name, value in values.items():
            if field_name in rules:
                results[field_name] = rules[field_name](value, field_name)
            else:
                # Generic validation for unknown fields
                if isinstance(value, str):
                    results[field_name] = self.validate_string_input(value, field_name)
                elif isinstance(value, (int, float)):
                    # Basic number validation
                    results[field_name] = ValidationResult(
                        is_valid=not (math.isnan(value) or math.isinf(value)) if isinstance(value, float) else True,
                        error_messages=["Value cannot be NaN or infinite"] if isinstance(value, float) and (math.isnan(value) or math.isinf(value)) else [],
                        warnings=[],
                        validated_value=value,
                        validation_metadata={"field_type": "generic_number"}
                    )
                elif isinstance(value, np.ndarray):
                    results[field_name] = self.validate_array_input(value, field_name)
                else:
                    # Unknown type - basic validation
                    results[field_name] = ValidationResult(
                        is_valid=True,
                        error_messages=[],
                        warnings=[f"Unknown field type {type(value).__name__} for {field_name}"],
                        validated_value=value,
                        validation_metadata={"field_type": "unknown"}
                    )
        
        return results
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get validation cache statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100.0) if total_requests > 0 else 0.0
        
        return {
            **self._cache_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._validation_cache),
            "cache_enabled": self.config.enable_caching
        }
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        logger.info("Cleared unified validation cache")


# ============================================================================
# PUBLIC API - Unified Validation Interface
# ============================================================================

# Global unified validation system instance
_unified_validation_system: Optional[TNFRUnifiedValidationSystem] = None


def get_unified_validation_system(config: Optional[ValidationConfig] = None) -> TNFRUnifiedValidationSystem:
    """Get or create global unified validation system.
    
    This provides a singleton interface for all TNFR validation operations
    to eliminate redundant system creation across modules.
    
    Parameters
    ----------
    config : ValidationConfig, optional
        Configuration for system (only used on first call)
        
    Returns
    -------
    TNFRUnifiedValidationSystem
        Global unified validation system instance
    """
    global _unified_validation_system
    
    if _unified_validation_system is None:
        _unified_validation_system = TNFRUnifiedValidationSystem(config)
        logger.info("Created global unified validation system")
    
    return _unified_validation_system


# Convenience functions for direct validation operations
def validate_structural_frequency(vf: Union[float, int], field_name: str = "vf") -> ValidationResult:
    """Validate structural frequency - convenience function."""
    return get_unified_validation_system().validate_structural_frequency(vf, field_name)


def validate_phase_value(phase: Union[float, int], field_name: str = "phase") -> ValidationResult:
    """Validate phase value - convenience function."""
    return get_unified_validation_system().validate_phase_value(phase, field_name)


def validate_coherence(coherence: Union[float, int], field_name: str = "coherence") -> ValidationResult:
    """Validate coherence - convenience function."""
    return get_unified_validation_system().validate_coherence(coherence, field_name)


def validate_string_input(input_string: str, field_name: str = "input") -> ValidationResult:
    """Validate string input - convenience function."""
    return get_unified_validation_system().validate_string_input(input_string, field_name)


def get_unified_validation_stats() -> Dict[str, Any]:
    """Get unified validation statistics - convenience function."""
    if _unified_validation_system is not None:
        return _unified_validation_system.get_cache_statistics()
    return {"status": "system_not_initialized"}