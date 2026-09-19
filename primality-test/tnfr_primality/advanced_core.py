"""Optional adapters to repository arithmetic helpers with basic fallbacks.

These paths compute divisor and factor statistics, then static arithmetic
pressure with unit default coefficients. Imported infrastructure availability
does not establish that every backend, cache or field reader is exercised.
No autonomous NFR evolution or full tetrad trajectory is produced by the
primality predicate. Returned certificates record arithmetic zero-pressure
checks; see the README for their conditional return type and scope."""

from __future__ import annotations

import math
import sys
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from .constants import (
    DELTA_NFR_THRESHOLD,
    ETA_CANONICAL,
    PRIMALITY_TOLERANCE,
    THETA_CANONICAL,
    ZETA_CANONICAL,
)

# Advanced imports - graceful fallback if not available
HAS_TNFR_INFRASTRUCTURE = False
_infrastructure_status = []

try:
    # Core TNFR repository infrastructure
    from tnfr.mathematics.number_theory import (
        ArithmeticStructuralTerms,
        ArithmeticTNFRFormalism,
        ArithmeticTNFRNetwork,
        ArithmeticTNFRParameters,
        PrimeCertificate,
    )

    _infrastructure_status.append("✓ Number theory engine")
except ImportError:
    _infrastructure_status.append("✗ Number theory engine")

try:
    from tnfr.constants.canonical import (
        GAMMA,
        MATH_DELTA_NFR_THRESHOLD_CANONICAL,
        PHI,
        PI,
        E,
    )

    _infrastructure_status.append("✓ Canonical constants")
except ImportError:
    _infrastructure_status.append("✗ Canonical constants")
    # Fallback constants
    PHI = 1.618033988749895
    GAMMA = 0.5772156649015329
    PI = 3.141592653589793
    E = 2.718281828459045
    MATH_DELTA_NFR_THRESHOLD_CANONICAL = 1e-12

try:
    from tnfr.utils.cache import (
        CacheLevel,
        TNFRHierarchicalCache,
        cache_tnfr_computation,
    )

    _infrastructure_status.append("✓ Advanced caching system")
except ImportError:
    _infrastructure_status.append("✗ Advanced caching system")

try:
    from tnfr.mathematics import get_backend

    _infrastructure_status.append("✓ Multi-backend mathematics")
except ImportError:
    _infrastructure_status.append("✗ Multi-backend mathematics")

try:
    from tnfr.physics.fields import (
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        estimate_coherence_length,
    )

    _infrastructure_status.append("✓ Structural field analysis")
except ImportError:
    _infrastructure_status.append("✗ Structural field analysis")

# Check if we have enough infrastructure
HAS_TNFR_INFRASTRUCTURE = any(
    "✓ Number theory engine" in s for s in _infrastructure_status
)


def get_infrastructure_status() -> str:
    """Get detailed infrastructure availability status."""
    status = "TNFR Advanced Infrastructure Status:\n"
    for item in _infrastructure_status:
        status += f"  {item}\n"
    status += f"\nAdvanced algorithms: {'ENABLED' if HAS_TNFR_INFRASTRUCTURE else 'DISABLED (fallback mode)'}"
    return status


# Shared unit defaults; a normalization choice, not a constants derivation.
ZETA = ZETA_CANONICAL
ETA = ETA_CANONICAL
THETA = THETA_CANONICAL
# Zero-detection tolerance for primality (abs(ΔNFR) ≤ TOLERANCE → prime)
TOLERANCE = PRIMALITY_TOLERANCE

# Global cache for expensive operations
_COMPUTATION_CACHE: Optional[Any] = None


def get_cache():
    """Get or create computation cache."""
    global _COMPUTATION_CACHE
    if _COMPUTATION_CACHE is None:
        if HAS_TNFR_INFRASTRUCTURE:
            try:
                _COMPUTATION_CACHE = TNFRHierarchicalCache(max_memory_mb=64)
            except:
                _COMPUTATION_CACHE = {}
        else:
            _COMPUTATION_CACHE = {}
    return _COMPUTATION_CACHE


def divisor_count_advanced(n: int) -> int:
    """Count divisors using a repository helper or enumeration fallback.

    Args:
        n: Positive integer.

    Returns:
        Divisor count. The repository branch constructs an arithmetic network
        per call and may fall back; no universal O(log n) runtime is claimed."""
    if n <= 0:
        raise ValueError("n must be positive")

    if HAS_TNFR_INFRASTRUCTURE:
        try:
            # Use advanced TNFR network computation with optimized sieve
            network = ArithmeticTNFRNetwork(max_number=max(100, min(n, 10000)))
            return network._divisor_count(n)
        except Exception:
            pass  # Fallback to basic implementation

    # Optimized fallback implementation
    count = 0
    sqrt_n = int(math.sqrt(n))

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            count += 1
            if i != n // i:  # Avoid counting square root twice
                count += 1

    return count


def divisor_sum_advanced(n: int) -> int:
    """Sum divisors using advanced TNFR algorithms.

    Args:
        n: Positive integer

    Returns:
        Sum of all divisors of n
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if HAS_TNFR_INFRASTRUCTURE:
        try:
            network = ArithmeticTNFRNetwork(max_number=max(100, min(n, 10000)))
            return network._divisor_sum(n)
        except Exception:
            pass

    # Optimized fallback
    sum_div = 0
    sqrt_n = int(math.sqrt(n))

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            sum_div += i
            if i != n // i:  # Avoid counting square root twice
                sum_div += n // i

    return sum_div


def prime_factor_count_advanced(n: int) -> int:
    """Count prime factors with advanced sieve algorithms.

    Args:
        n: Positive integer

    Returns:
        Number of prime factors counting multiplicity
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 0

    if HAS_TNFR_INFRASTRUCTURE:
        try:
            network = ArithmeticTNFRNetwork(max_number=max(100, min(n, 10000)))
            return network._prime_factor_count(n)
        except Exception:
            pass

    # Optimized fallback
    count = 0

    # Handle factor 2
    while n % 2 == 0:
        count += 1
        n //= 2

    # Handle odd factors
    d = 3
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 2

    # If n is still > 1, it's a prime factor
    if n > 1:
        count += 1

    return count


def tnfr_delta_nfr_advanced(
    n: int,
    *,
    zeta: float = ZETA,
    eta: float = ETA,
    theta: float = THETA,
    use_cache: bool = True,
) -> float:
    """Compute static arithmetic pressure through an optional repository adapter.

    Args:
        n: Integer; values below 2 return positive infinity.
        zeta: Multiplicity pressure coefficient, default 1.0.
        eta: Divisor pressure coefficient, default 1.0.
        theta: Abundance pressure coefficient, default 1.0.
        use_cache: Retained compatibility parameter; this body does not use it.

    Returns:
        Floating pressure from supplied arithmetic statistics. Its exact-real
        prime zero-set statement requires positive coefficients; the fallback
        does not validate that premise. Cached wrappers are separate functions."""
    if n <= 1:
        return float("inf")  # Not prime by definition

    if HAS_TNFR_INFRASTRUCTURE:
        try:
            # Use advanced TNFR formalism with optimized parameters
            params = ArithmeticTNFRParameters(zeta=zeta, eta=eta, theta=theta)

            # Compute structural terms using optimized functions
            omega_n = prime_factor_count_advanced(n)
            tau_n = divisor_count_advanced(n)
            sigma_n = divisor_sum_advanced(n)
            terms = ArithmeticStructuralTerms(tau=tau_n, sigma=sigma_n, omega=omega_n)

            # Use canonical formalism with component breakdown
            return ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
        except Exception:
            pass  # Fallback to basic computation

    # Fallback computation
    omega_n = prime_factor_count_advanced(n)
    tau_n = divisor_count_advanced(n)
    sigma_n = divisor_sum_advanced(n)

    # TNFR arithmetic pressure equation
    factorization_pressure = zeta * (omega_n - 1)
    divisor_pressure = eta * (tau_n - 2)
    sigma_pressure = theta * (sigma_n / n - (1 + 1 / n))

    return factorization_pressure + divisor_pressure + sigma_pressure


def tnfr_is_prime_advanced(n: int, *, return_certificate: bool = False):
    """Return an arithmetic predicate or an optional arithmetic certificate.

    Args:
        n: Integer to test.
        return_certificate: Request a repository ``PrimeCertificate`` when
            infrastructure is available and the certificate call succeeds.

    Returns:
        Usually ``(is_prime, pressure)``. A successful certificate request can
        return a ``PrimeCertificate`` instead. Inputs n<=1 and n=2, unavailable
        infrastructure, or a caught certificate failure still return a tuple.
        Callers must inspect the actual type rather than assume the request
        guarantees a certificate.

    The result is a check on divisor/factor statistics and a numerical pressure
    tolerance. It is not a full nodal triad, operator trace or dynamics proof."""
    if n <= 1:
        return False, float("inf")
    if n == 2:
        return True, 0.0

    if HAS_TNFR_INFRASTRUCTURE and return_certificate:
        try:
            # Request the arithmetic-statistic certificate; no operator trace.
            params = ArithmeticTNFRParameters()

            omega_n = prime_factor_count_advanced(n)
            tau_n = divisor_count_advanced(n)
            sigma_n = divisor_sum_advanced(n)
            terms = ArithmeticStructuralTerms(tau=tau_n, sigma=sigma_n, omega=omega_n)

            certificate = ArithmeticTNFRFormalism.prime_certificate(
                n, terms, params, tolerance=TOLERANCE
            )

            return certificate  # Return arithmetic certificate object
        except Exception:
            pass

    if HAS_TNFR_INFRASTRUCTURE:
        try:
            # Use basic advanced computation without certificate
            params = ArithmeticTNFRParameters()

            omega_n = prime_factor_count_advanced(n)
            tau_n = divisor_count_advanced(n)
            sigma_n = divisor_sum_advanced(n)
            terms = ArithmeticStructuralTerms(tau=tau_n, sigma=sigma_n, omega=omega_n)

            certificate = ArithmeticTNFRFormalism.prime_certificate(
                n, terms, params, tolerance=TOLERANCE
            )

            return certificate.structural_prime, certificate.delta_nfr
        except Exception:
            pass

    # Fallback computation
    delta_nfr = tnfr_delta_nfr_advanced(n)
    is_prime = abs(delta_nfr) <= TOLERANCE

    return is_prime, delta_nfr


# Advanced caching decorator if infrastructure available
if HAS_TNFR_INFRASTRUCTURE:
    try:

        @cache_tnfr_computation(
            level=CacheLevel.DERIVED_METRICS,
            dependencies={"arithmetic_properties"},
            cost_estimator=lambda n: math.log(max(n, 2)),
        )
        def cached_tnfr_is_prime_advanced(n: int):
            """Hierarchically cached TNFR primality test with dependency tracking."""
            return tnfr_is_prime_advanced(n)

    except Exception:
        # Fallback to LRU if TNFR caching fails
        @lru_cache(maxsize=1024)
        def cached_tnfr_is_prime_advanced(n: int):
            """LRU cached TNFR primality test."""
            return tnfr_is_prime_advanced(n)

else:
    # Fallback LRU cache
    @lru_cache(maxsize=1024)
    def cached_tnfr_is_prime_advanced(n: int):
        """LRU cached TNFR primality test."""
        return tnfr_is_prime_advanced(n)


def validate_tnfr_theory_advanced(max_n: int = 1000) -> Dict[str, Any]:
    """Collect finite arithmetic validation and timing diagnostics.

    The report compares selected primality implementations and records optional
    analysis where available. This is not a proof of general runtime accuracy,
    network dynamics or emergent physical behavior.

    Args:
        max_n: Maximum tested integer, default 1000.

    Returns:
        Statistics and optional diagnostic summaries for that finite input set."""
    start_time = time.time()

    results: Dict[str, Any] = {
        "tested_numbers": 0,
        "correct_predictions": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "accuracy": 0.0,
        "prime_examples": [],
        "composite_examples": [],
        "performance_ms": 0.0,
        "infrastructure_used": HAS_TNFR_INFRASTRUCTURE,
        "infrastructure_status": _infrastructure_status,
        "algorithm_version": "advanced",
    }

    if HAS_TNFR_INFRASTRUCTURE:
        try:
            # Use advanced network validation with full analytics
            network = ArithmeticTNFRNetwork(max_number=max_n)

            # Get comprehensive network statistics
            stats = network.summary_statistics()
            results.update(
                {
                    "network_statistics": stats,
                    "prime_mean_delta_nfr": stats.get("prime_mean_DELTA_NFR", 0.0),
                    "composite_mean_delta_nfr": stats.get(
                        "composite_mean_DELTA_NFR", 0.0
                    ),
                    "total_primes_found": stats.get("prime_count", 0),
                    "prime_ratio": stats.get("prime_ratio", 0.0),
                }
            )

            # Analyze prime characteristics with advanced metrics
            try:
                prime_chars = network.analyze_prime_characteristics()
                results["prime_characteristics"] = prime_chars
            except Exception:
                pass

            # Detect numerical zero-pressure candidates using the configured tolerance
            try:
                candidates = network.detect_prime_candidates(
                    delta_nfr_threshold=TOLERANCE, return_certificates=True
                )
                results["candidate_count"] = len(candidates)
            except Exception:
                pass

            # Performance comparison tests
            performance_tests = []

            # Test a sample of numbers for detailed analysis
            test_numbers = [997, 1009, 1013, 1019, 1021]  # Mix of primes and composites
            for test_n in test_numbers:
                if test_n <= max_n:
                    perf_start = time.time()
                    is_prime, delta_nfr = cached_tnfr_is_prime_advanced(test_n)
                    perf_time = (time.time() - perf_start) * 1000

                    performance_tests.append(
                        {
                            "n": test_n,
                            "is_prime": is_prime,
                            "delta_nfr": delta_nfr,
                            "time_ms": perf_time,
                        }
                    )

            results["performance_tests"] = performance_tests

        except Exception as e:
            results["infrastructure_error"] = str(e)
            # Continue with fallback validation

    # Run validation on known primes up to max_n
    known_primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
        307,
        311,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439,
        443,
        449,
        457,
        461,
        463,
        467,
        479,
        487,
        491,
        499,
        503,
        509,
        521,
        523,
        541,
        547,
        557,
        563,
        569,
        571,
        577,
        587,
        593,
        599,
        601,
        607,
        613,
        617,
        619,
        631,
        641,
        643,
        647,
        653,
        659,
        661,
        673,
        677,
        683,
        691,
        701,
        709,
        719,
        727,
        733,
        739,
        743,
        751,
        757,
        761,
        769,
        773,
        787,
        797,
        809,
        811,
        821,
        823,
        827,
        829,
        839,
        853,
        857,
        859,
        863,
        877,
        881,
        883,
        887,
        907,
        911,
        919,
        929,
        937,
        941,
        947,
        953,
        967,
        971,
        977,
        983,
        991,
        997,
    ]

    known_primes_set = set(p for p in known_primes if p <= max_n)

    # Component analysis
    delta_nfr_distribution = []

    for n in range(2, min(max_n + 1, 1001)):  # Limit to 1000 for performance
        results["tested_numbers"] += 1

        is_known_prime = n in known_primes_set
        tnfr_prime, delta_nfr = cached_tnfr_is_prime_advanced(n)

        delta_nfr_distribution.append(delta_nfr)

        if is_known_prime == tnfr_prime:
            results["correct_predictions"] += 1
        elif tnfr_prime and not is_known_prime:
            results["false_positives"] += 1
        elif not tnfr_prime and is_known_prime:
            results["false_negatives"] += 1

        # Store detailed examples
        if is_known_prime and len(results["prime_examples"]) < 10:
            example = {"n": int(n), "delta_nfr": float(delta_nfr)}

            # Add certificate data if available
            if HAS_TNFR_INFRASTRUCTURE:
                try:
                    cert = tnfr_is_prime_advanced(n, return_certificate=True)
                    if hasattr(cert, "explanation"):
                        example["certificate"] = {
                            "tau": int(cert.tau) if hasattr(cert, "tau") else 0,
                            "sigma": (
                                float(cert.sigma) if hasattr(cert, "sigma") else 0.0
                            ),
                            "omega": int(cert.omega) if hasattr(cert, "omega") else 0,
                            "explanation": (
                                str(cert.explanation)
                                if hasattr(cert, "explanation")
                                else ""
                            ),
                        }
                except Exception:
                    pass

            results["prime_examples"].append(example)

        elif not is_known_prime and len(results["composite_examples"]) < 10:
            results["composite_examples"].append(
                {"n": int(n), "delta_nfr": float(delta_nfr)}
            )

    # Calculate final metrics
    if results["tested_numbers"] > 0:
        results["accuracy"] = results["correct_predictions"] / results["tested_numbers"]

    # Statistical analysis of ΔNFR distribution
    if delta_nfr_distribution:
        results["delta_nfr_stats"] = {
            "mean": float(sum(delta_nfr_distribution) / len(delta_nfr_distribution)),
            "min": float(min(delta_nfr_distribution)),
            "max": float(max(delta_nfr_distribution)),
            "count": int(len(delta_nfr_distribution)),
        }

    perf_ms = float((time.time() - start_time) * 1000)
    results["performance_ms"] = perf_ms
    tested_count = results["tested_numbers"]
    results["numbers_per_second"] = (
        float(tested_count / (perf_ms / 1000)) if perf_ms > 0 else 0.0
    )

    return results


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system and infrastructure information."""
    return {
        "python_version": sys.version,
        "infrastructure_available": HAS_TNFR_INFRASTRUCTURE,
        "infrastructure_status": _infrastructure_status,
        "cache_available": _COMPUTATION_CACHE is not None,
        "constants": {
            "zeta": ZETA,
            "eta": ETA,
            "theta": THETA,
            "tolerance": TOLERANCE,
            "phi": PHI,
            "gamma": GAMMA,
            "pi": PI,
            "e": E,
        },
    }
