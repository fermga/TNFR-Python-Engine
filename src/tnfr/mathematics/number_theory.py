
"""
TNFR Arithmetic Network: Prime Numbers as Structural Attractors

Implementation of the theoretical framework for detecting prime emergence
from TNFR structural dynamics applied to natural numbers.

Author: TNFR Research Team
Date: 2025-11-13
Status: IMPLEMENTATION PROTOTYPE
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# Centralized TNFR cache infrastructure (robust, shared across repo)
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel  # type: ignore
    _CACHE_OK = True
except Exception:  # Fallback if utils.cache not available in this context
    _CACHE_OK = False

    def cache_tnfr_computation(*args, **kwargs):
        def _deco(f):
            return f
        return _deco

    class CacheLevel:
        DERIVED_METRICS = None

# Import centralized TNFR physics functions for maximum reuse and cache efficiency
try:
    from ..physics.fields import (
        compute_phase_gradient,
        compute_phase_curvature,
        compute_structural_potential as centralized_phi_s,
        estimate_coherence_length as centralized_xi_c
    )
    HAS_CENTRALIZED_FIELDS = True
except ImportError:
    HAS_CENTRALIZED_FIELDS = False

# Local imports (will integrate with existing TNFR modules)
try:
    from sympy import divisor_count, divisor_sigma, factorint, isprime
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    logger.warning(" sympy not available. Using basic implementations.")


# ============================================================================
# ARITHMETIC TNFR NETWORK CLASS
# ============================================================================

@dataclass
class ArithmeticTNFRParameters:
    """Calibration parameters for arithmetic TNFR system."""

    # EPI parameters
    alpha: float = 0.5    # Weight for factorization complexity
    beta: float = 0.3     # Weight for divisor complexity
    gamma: float = 0.2    # Weight for divisor excess

    # Frequency parameters
    nu_0: float = 1.0     # Base arithmetic frequency
    delta: float = 0.1    # Divisor density weight
    epsilon: float = 0.05  # Factorization complexity weight

    # Pressure parameters
    zeta: float = 1.0     # Factorization pressure weight
    eta: float = 0.8      # Divisor pressure weight
    theta: float = 0.6    # Sigma pressure weight


@dataclass(frozen=True)
class ArithmeticStructuralTerms:
    """Canonical arithmetic invariants per natural number node."""

    tau: int
    sigma: int
    omega: int

    def as_dict(self) -> Dict[str, int]:
        return {'tau': self.tau, 'sigma': self.sigma, 'omega': self.omega}


@dataclass(frozen=True)
class PrimeCertificate:
    """Structured report for the TNFR prime criterion ΔNFR = 0."""

    number: int
    delta_nfr: float
    structural_prime: bool
    tolerance: float
    tau: int
    sigma: int
    omega: int
    explanation: str
    components: Optional[Dict[str, float]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            'number': self.number,
            'delta_nfr': self.delta_nfr,
            'structural_prime': self.structural_prime,
            'tolerance': self.tolerance,
            'tau': self.tau,
            'sigma': self.sigma,
            'omega': self.omega,
            'components': dict(self.components) if self.components is not None else None,
            'explanation': self.explanation,
        }


class ArithmeticTNFRFormalism:
    """Explicit formulas that tie TNFR physics to arithmetic invariants."""

    @staticmethod
    def epi_value(n: int, terms: ArithmeticStructuralTerms, params: ArithmeticTNFRParameters) -> float:
        divisor_complexity = params.beta * math.log(max(terms.tau, 1))
        divisor_excess = params.gamma * (terms.sigma / n - 1)
        factorization_complexity = params.alpha * terms.omega
        return 1.0 + factorization_complexity + divisor_complexity + divisor_excess

    @staticmethod
    def frequency_value(n: int, terms: ArithmeticStructuralTerms, params: ArithmeticTNFRParameters) -> float:
        divisor_density = params.delta * terms.tau / n
        factorization_term = params.epsilon * terms.omega / math.log(n)
        return params.nu_0 * (1.0 + divisor_density + factorization_term)

    @staticmethod
    def delta_nfr_value(n: int, terms: ArithmeticStructuralTerms, params: ArithmeticTNFRParameters) -> float:
        factorization_pressure = params.zeta * (terms.omega - 1)
        divisor_pressure = params.eta * (terms.tau - 2)
        sigma_pressure = params.theta * (terms.sigma / n - (1 + 1 / n))
        return factorization_pressure + divisor_pressure + sigma_pressure

    @staticmethod
    def component_breakdown(n: int, terms: ArithmeticStructuralTerms, params: ArithmeticTNFRParameters) -> Dict[str, float]:
        return {
            'factorization_pressure': params.zeta * (terms.omega - 1),
            'divisor_pressure': params.eta * (terms.tau - 2),
            'sigma_pressure': params.theta * (terms.sigma / n - (1 + 1 / n)),
        }

    @staticmethod
    def local_coherence(delta_nfr: float) -> float:
        return 1.0 / (1.0 + abs(delta_nfr))

    @staticmethod
    def symbolic_delta_nfr(params: Optional[ArithmeticTNFRParameters] = None):
        params = params or ArithmeticTNFRParameters()
        try:
            import sympy as sp  # type: ignore

            omega, tau, sigma, n = sp.symbols('omega tau sigma n', positive=True)
            expr = (
                params.zeta * (omega - 1)
                + params.eta * (tau - 2)
                + params.theta * (sigma / n - (1 + 1 / n))
            )
            return expr
        except Exception:
            return (
                f"ΔNFR(n)= {params.zeta}(ω-1) + {params.eta}(τ-2) + "
                f"{params.theta}(σ/n - (1+1/n))"
            )

    @staticmethod
    def prime_certificate(
        n: int,
        terms: ArithmeticStructuralTerms,
        params: ArithmeticTNFRParameters,
        *,
        tolerance: float = 1e-12,
        components: Optional[Dict[str, float]] = None,
    ) -> PrimeCertificate:
        if components is None:
            components = ArithmeticTNFRFormalism.component_breakdown(n, terms, params)
        delta = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
        structural_prime = abs(delta) <= tolerance
        explanation = (
            "ΔNFR vanishes within tolerance; node is a structural attractor"
            if structural_prime else
            "ΔNFR ≠ 0, coherence pressure reveals composite structure"
        )
        return PrimeCertificate(
            number=n,
            delta_nfr=float(delta),
            structural_prime=structural_prime,
            tolerance=float(tolerance),
            tau=terms.tau,
            sigma=terms.sigma,
            omega=terms.omega,
            components=components,
            explanation=explanation,
        )


class ArithmeticTNFRNetwork:
    """
    TNFR network where nodes are natural numbers and dynamics reveal prime structure.

    Each number n becomes a TNFR node with:
    - EPI_n: Arithmetic structural form
    - νf_n: Arithmetic frequency
    - ΔNFR_n: Factorization pressure

    Prime numbers should emerge as structural attractors (ΔNFR ≈ 0).
    """
    
    def __init__(self, max_number: int = 100, parameters: Optional[ArithmeticTNFRParameters] = None):
        """
        Initialize arithmetic TNFR network.
        
        Args:
            max_number: Maximum number to include in network (default 100)
            parameters: TNFR calibration parameters
        """
        self.max_number = max_number
        self.params = parameters or ArithmeticTNFRParameters()
        
        # Build the network
        self.graph = self._construct_arithmetic_network()
        self._graph_undirected_cache: Optional[nx.Graph] = None
        self._compute_tnfr_properties()
        
    def _construct_arithmetic_network(self) -> nx.DiGraph:
        """Construct directed graph representing arithmetic relationships."""
        G = nx.DiGraph()

        # Add number nodes
        for n in range(2, self.max_number + 1):  # Start from 2 (first prime)
            G.add_node(n, number=n)

        # Add edges based on arithmetic relationships
        for n1 in range(2, self.max_number + 1):
            for n2 in range(n1 + 1, self.max_number + 1):

                # Divisibility links
                if n2 % n1 == 0:  # n1 divides n2
                    weight = self._divisibility_weight(n1, n2)
                    G.add_edge(n1, n2, weight=weight, type='divisibility')

                # GCD links (for numbers sharing factors)
                gcd_val = math.gcd(n1, n2)
                if gcd_val > 1:
                    weight = self._gcd_weight(n1, n2, gcd_val)
                    G.add_edge(n1, n2, weight=weight, type='gcd')
                    G.add_edge(n2, n1, weight=weight, type='gcd')  # Symmetric

        return G
    
    def _divisibility_weight(self, n1: int, n2: int) -> float:
        """Compute link weight for divisibility relationship."""
        if n2 % n1 != 0:
            return 0.0
        quotient = n2 // n1
        return 1.0 / math.log(quotient + 1)
    
    def _gcd_weight(self, n1: int, n2: int, gcd_val: int) -> float:
        """Compute link weight based on GCD."""
        return gcd_val / max(n1, n2)
    
    def _compute_tnfr_properties(self) -> None:
        """Compute TNFR properties (EPI, νf, ΔNFR) for each number."""
        for n in self.graph.nodes():
            # Compute arithmetic functions
            tau_n = self._divisor_count(n)
            sigma_n = self._divisor_sum(n)
            omega_n = self._prime_factor_count(n)
            terms = ArithmeticStructuralTerms(tau=tau_n, sigma=sigma_n, omega=omega_n)
            
            # Compute TNFR properties via the formalism helpers
            epi_n = ArithmeticTNFRFormalism.epi_value(n, terms, self.params)
            nu_f_n = ArithmeticTNFRFormalism.frequency_value(n, terms, self.params)
            delta_nfr_n = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, self.params)
            local_coherence = ArithmeticTNFRFormalism.local_coherence(delta_nfr_n)
            component_pressures = ArithmeticTNFRFormalism.component_breakdown(n, terms, self.params)

            # Store TNFR telemetry for this number node
            self.graph.nodes[n].update({
                'tau': tau_n,            # Number of divisors
                'sigma': sigma_n,        # Sum of divisors
                'omega': omega_n,        # Prime factor count (with multiplicity)
                'EPI': epi_n,            # Structural form
                'nu_f': nu_f_n,          # Structural frequency
                'DELTA_NFR': delta_nfr_n,  # Factorization pressure
                'is_prime': self._is_prime(n),  # Ground truth for validation
                'structural_terms': terms,
                'delta_components': component_pressures,
                'coherence_local': local_coherence,
            })
    
    # ========================================================================
    # NUMBER THEORY FUNCTIONS
    # ========================================================================
    
    def _divisor_count(self, n: int) -> int:
        """Count the number of divisors of n."""
        if HAS_SYMPY:
            return int(divisor_count(n))
        else:
            # Basic implementation
            count = 0
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:  # Avoid counting square root twice
                        count += 1
            return count
    
    def _divisor_sum(self, n: int) -> int:
        """Sum of all divisors of n."""
        if HAS_SYMPY:
            return int(divisor_sigma(n, 1))
        else:
            # Basic implementation
            total = 0
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    total += i
                    if i != n // i:
                        total += n // i
            return total
    
    def _prime_factor_count(self, n: int) -> int:
        """Count prime factors with multiplicity (Ω(n))."""
        if HAS_SYMPY:
            return sum(factorint(n).values())
        else:
            # Basic implementation
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if HAS_SYMPY:
            return isprime(n)
        else:
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
    
    # ========================================================================
    # PRIME DETECTION AND ANALYSIS
    # ========================================================================
    
    def detect_prime_candidates(
        self,
        delta_nfr_threshold: float = 0.1,
        *,
        tolerance: float = 1e-12,
        return_certificates: bool = False,
    ) -> List[Union[Tuple[int, float], PrimeCertificate]]:
        """
        Detect numbers that behave like primes (low ΔNFR).
        
        Args:
            delta_nfr_threshold: Maximum ΔNFR for prime candidates
            tolerance: Absolute tolerance for ΔNFR when returning certificates
            return_certificates: When True, return full PrimeCertificate objects
            
        Returns:
            List of (number, ΔNFR) pairs for prime candidates
        """
        candidates = []
        
        for n in self.graph.nodes():
            delta_nfr = self.graph.nodes[n]['DELTA_NFR']
            if abs(delta_nfr) <= delta_nfr_threshold:
                if return_certificates:
                    candidates.append(self.get_prime_certificate(n, tolerance=tolerance))
                else:
                    candidates.append((n, delta_nfr))
                
        # Sort by ΔNFR (most stable first)
        if return_certificates:
            candidates.sort(key=lambda cert: abs(cert.delta_nfr))
        else:
            candidates.sort(key=lambda x: abs(x[1]))
        return candidates

    def get_structural_terms(self, n: int) -> ArithmeticStructuralTerms:
        """Return the canonical structural terms (τ, σ, ω) for node n."""
        if n not in self.graph.nodes:
            raise ValueError(f"Number {n} not in network (max: {self.max_number})")
        terms = self.graph.nodes[n].get('structural_terms')
        if isinstance(terms, ArithmeticStructuralTerms):
            return terms
        # Reconstruct if older cache stored dicts
        return ArithmeticStructuralTerms(
            tau=int(self.graph.nodes[n]['tau']),
            sigma=int(self.graph.nodes[n]['sigma']),
            omega=int(self.graph.nodes[n]['omega']),
        )

    def get_delta_components(self, n: int) -> Dict[str, float]:
        """Return component-level contributions to ΔNFR for node n."""
        if n not in self.graph.nodes:
            raise ValueError(f"Number {n} not in network (max: {self.max_number})")
        components = self.graph.nodes[n].get('delta_components')
        if components is None:
            terms = self.get_structural_terms(n)
            components = ArithmeticTNFRFormalism.component_breakdown(n, terms, self.params)
            self.graph.nodes[n]['delta_components'] = components
        return dict(components)

    def get_prime_certificate(
        self,
        n: int,
        *,
        tolerance: float = 1e-12,
        include_components: bool = True,
    ) -> PrimeCertificate:
        """Generate a PrimeCertificate using the stored TNFR telemetry."""
        if n not in self.graph.nodes:
            raise ValueError(f"Number {n} not in network (max: {self.max_number})")
        terms = self.get_structural_terms(n)
        components = self.get_delta_components(n) if include_components else None
        return ArithmeticTNFRFormalism.prime_certificate(
            n,
            terms,
            self.params,
            tolerance=tolerance,
            components=components,
        )

    def generate_prime_certificates(
        self,
        numbers: Optional[Iterable[int]] = None,
        *,
        tolerance: float = 1e-12,
        include_components: bool = True,
    ) -> List[PrimeCertificate]:
        """Return PrimeCertificates for the provided numbers (or all nodes)."""
        if numbers is None:
            numbers = list(self.graph.nodes())
        certificates: List[PrimeCertificate] = []
        for n in numbers:
            if n in self.graph.nodes:
                certificates.append(
                    self.get_prime_certificate(
                        n,
                        tolerance=tolerance,
                        include_components=include_components,
                    )
                )
        certificates.sort(key=lambda cert: cert.number)
        return certificates
    
    def validate_prime_detection(self, delta_nfr_threshold: float = 0.1) -> Dict[str, float]:
        """
        Validate TNFR prime detection against known primes.
        
        Returns:
            Dictionary with precision, recall, F1-score
        """
        candidates = self.detect_prime_candidates(delta_nfr_threshold)
        predicted_primes = {x[0] for x in candidates}
        
        actual_primes = {n for n in self.graph.nodes() if self.graph.nodes[n]['is_prime']}
        
        # Compute metrics
        true_positives = len(predicted_primes & actual_primes)
        false_positives = len(predicted_primes - actual_primes)
        false_negatives = len(actual_primes - predicted_primes)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'predicted_primes': sorted(predicted_primes),
            'actual_primes': sorted(actual_primes),
            'missed_primes': sorted(actual_primes - predicted_primes),
            'false_alarms': sorted(predicted_primes - actual_primes)
        }
    
    def get_tnfr_properties(self, n: int) -> Dict[str, float]:
        """Get all TNFR properties for a specific number."""
        if n not in self.graph.nodes():
            raise ValueError(f"Number {n} not in network (max: {self.max_number})")
            
        node_data = self.graph.nodes[n]
        return {
            'number': n,
            'tau': node_data['tau'],
            'sigma': node_data['sigma'],
            'omega': node_data['omega'],
            'EPI': node_data['EPI'],
            'nu_f': node_data['nu_f'],
            'DELTA_NFR': node_data['DELTA_NFR'],
            'is_prime': node_data['is_prime'],
            'structural_terms': node_data['structural_terms'].as_dict() if isinstance(node_data.get('structural_terms'), ArithmeticStructuralTerms) else node_data.get('structural_terms'),
            'delta_components': dict(node_data['delta_components']) if node_data.get('delta_components') is not None else None,
            'coherence_local': node_data.get('coherence_local'),
        }
    
    def analyze_prime_characteristics(self) -> Dict[str, List[float]]:
        """Analyze TNFR characteristics of all primes in the network."""
        primes = [n for n in self.graph.nodes() if self.graph.nodes[n]['is_prime']]
        
        characteristics = {
            'numbers': primes,
            'EPI_values': [],
            'nu_f_values': [],
            'DELTA_NFR_values': []
        }
        
        for p in primes:
            node_data = self.graph.nodes[p]
            characteristics['EPI_values'].append(node_data['EPI'])
            characteristics['nu_f_values'].append(node_data['nu_f'])
            characteristics['DELTA_NFR_values'].append(node_data['DELTA_NFR'])
            
        return characteristics
    
    def summary_statistics(self) -> Dict[str, float]:
        """Compute summary statistics for the entire network."""
        all_nodes = list(self.graph.nodes())
        primes = [n for n in all_nodes if self.graph.nodes[n]['is_prime']]
        composites = [n for n in all_nodes if not self.graph.nodes[n]['is_prime']]
        
        # Prime statistics
        prime_delta_nfr = [self.graph.nodes[p]['DELTA_NFR'] for p in primes]
        prime_epi = [self.graph.nodes[p]['EPI'] for p in primes]
        
    # Composite statistics
        composite_delta_nfr = [self.graph.nodes[c]['DELTA_NFR'] for c in composites]
        composite_epi = [self.graph.nodes[c]['EPI'] for c in composites]
        
        return {
            'total_numbers': len(all_nodes),
            'prime_count': len(primes),
            'composite_count': len(composites),
            'prime_ratio': len(primes) / len(all_nodes),
            
            # Prime characteristics
            'prime_mean_DELTA_NFR': np.mean(prime_delta_nfr) if prime_delta_nfr else 0,
            'prime_std_DELTA_NFR': np.std(prime_delta_nfr) if prime_delta_nfr else 0,
            'prime_mean_EPI': np.mean(prime_epi) if prime_epi else 0,
            
            # Composite characteristics
            'composite_mean_DELTA_NFR': np.mean(composite_delta_nfr) if composite_delta_nfr else 0,
            'composite_std_DELTA_NFR': np.std(composite_delta_nfr) if composite_delta_nfr else 0,
            'composite_mean_EPI': np.mean(composite_epi) if composite_epi else 0,
            
            # Separation metrics
            'DELTA_NFR_separation': (np.mean(composite_delta_nfr) - np.mean(prime_delta_nfr)) if (composite_delta_nfr and prime_delta_nfr) else 0,
            'EPI_separation': (np.mean(composite_epi) - np.mean(prime_epi)) if (composite_epi and prime_epi) else 0
        }

    # ========================================================================
    # STRUCTURAL FIELD TELEMETRY (Φ_s, |∇φ|, K_φ, ξ_C)
    # ========================================================================

    # Cached helpers (use centralized repo cache infra when available)
    @staticmethod
    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS if _CACHE_OK else None,
        dependencies={'DELTA_NFR', 'graph_structure'},
    )
    def _cached_phi_s_helper(
        G: nx.Graph,
        *,
        alpha: float = 2.0,
        distance_mode: str = "topological",
        dnfr_attr: str = 'DELTA_NFR',
    ) -> Dict[int, float]:
        phi_s: Dict[int, float] = {}
        if distance_mode == 'topological':
            spl = dict(nx.all_pairs_shortest_path_length(G))
            for i in G.nodes():
                acc = 0.0
                for j, dij in spl[i].items():
                    if i == j or dij == 0:
                        continue
                    acc += float(G.nodes[j].get(dnfr_attr, 0.0)) / (dij ** alpha)
                phi_s[i] = acc
            return phi_s

        # arithmetic distance
        nodes = sorted(G.nodes())
        delta = np.array([float(G.nodes[n].get(dnfr_attr, 0.0)) for n in nodes], dtype=float)
        x = np.arange(len(nodes))
        for i_idx, i in enumerate(nodes):
            dist = np.abs(x - i_idx)
            dist[i_idx] = 1
            save = delta[i_idx]
            delta[i_idx] = 0.0
            acc = float(np.sum(delta / (dist ** alpha)))
            delta[i_idx] = save
            phi_s[i] = acc
        return phi_s

    @staticmethod
    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS if _CACHE_OK else None,
        dependencies={'DELTA_NFR', 'graph_structure'},
    )
    def _cached_xi_c_helper(
        G: nx.Graph,
        *,
        min_pairs: int = 5,
        distance_mode: str = 'topological',
        dnfr_attr: str = 'DELTA_NFR',
    ) -> Dict[str, object]:
        c = {i: 1.0 / (1.0 + abs(float(G.nodes[i].get(dnfr_attr, 0.0)))) for i in G.nodes()}
        accum: Dict[int, List[float]] = {}
        nodes = sorted(G.nodes())
        if distance_mode == 'topological':
            spl = dict(nx.all_pairs_shortest_path_length(G))
            for i in nodes:
                for j, dist in spl[i].items():
                    if j <= i or dist <= 0:
                        continue
                    accum.setdefault(dist, []).append(c[i] * c[j])
        else:
            for a_idx, i in enumerate(nodes):
                for b_idx in range(a_idx + 1, len(nodes)):
                    j = nodes[b_idx]
                    dist = abs(i - j)
                    accum.setdefault(dist, []).append(c[i] * c[j])

        r_vals: List[int] = []
        C_vals: List[float] = []
        for r in sorted(accum.keys()):
            pairs = accum[r]
            if len(pairs) >= min_pairs:
                mv = float(np.mean(pairs))
                if mv > 0:
                    r_vals.append(r)
                    C_vals.append(mv)
        result: Dict[str, object] = {
            'r': r_vals,
            'C_r': C_vals,
            'xi_c': None,
            'fit_intercept': None,
            'fit_slope': None,
            'R2': None,
        }
        if len(C_vals) >= 2:
            y = np.log(np.array(C_vals))
            x = np.array(r_vals)
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
            y_pred = slope * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            xi_c = (-1.0 / slope) if slope < 0 else None
            result.update({
                'xi_c': xi_c,
                'fit_intercept': float(intercept),
                'fit_slope': float(slope),
                'R2': float(r2),
            })
        return result

    def _get_undirected_graph(self) -> nx.Graph:
        """Return an undirected view/cached copy for distance and neighbor ops."""
        if self._graph_undirected_cache is None:
            self._graph_undirected_cache = self.graph.to_undirected()
        return self._graph_undirected_cache

    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS if _CACHE_OK else None,
        dependencies={'graph_structure', 'nu_f'},
    )
    def _cached_compute_phase_helper(self, method: str = "spectral") -> Dict[int, float]:
        """Cached helper for phase computation to avoid recomputation."""
        G = self._get_undirected_graph()
        phi: Dict[int, float] = {}

        if method == "spectral":
            try:
                pos = nx.spectral_layout(G, dim=2)
                for n, (x, y) in pos.items():
                    theta = math.atan2(y, x)
                    if theta < 0:
                        theta += 2 * math.pi
                    phi[n] = theta
            except Exception:
                method = "logn"  # Fallback

        if method == "logn":
            denom = math.log(self.max_number + 1)
            for n in G.nodes():
                frac = (math.log(n) / denom) % 1.0
                phi[n] = 2 * math.pi * frac

        if method == "nuf":
            vals = [self.graph.nodes[i]['nu_f'] for i in G.nodes()]
            maxv = max(vals) if vals else 1.0
            for n in G.nodes():
                frac = (self.graph.nodes[n]['nu_f'] / maxv) % 1.0
                phi[n] = 2 * math.pi * frac

        return phi

    def compute_phase(self, method: str = "spectral", store: bool = True) -> Dict[int, float]:
        """
        Compute per-node phase φ ∈ [0, 2π).

        Methods:
        - spectral: angle from 2D spectral layout (graph Laplacian eigen-embedding)
        - logn: φ = 2π * frac(log(n)/log(max_n+1))
        - nuf: φ = 2π * frac(nu_f / max_nu_f)
        """
        # Use cached computation
        phi = self._cached_compute_phase_helper(method=method)
        
        if store:
            for n, v in phi.items():
                self.graph.nodes[n]['phi'] = v
        return phi

    def compute_phase_gradient(self) -> Dict[int, float]:
        """Compute |∇φ|(i) = mean_{j∈N(i)} |φ_i - φ_j| (neighbors in undirected graph)."""
        if any('phi' not in self.graph.nodes[i] for i in self.graph.nodes()):
            self.compute_phase(store=True)
        
        # Use centralized cached function from physics.fields (CANONICAL)
        if HAS_CENTRALIZED_FIELDS:
            G = self._get_undirected_graph()
            phi_grad = compute_phase_gradient(G)
        else:
            # Fallback implementation
            G = self._get_undirected_graph()
            phi_grad = {}
            for i in G.nodes():
                nbrs = list(G.neighbors(i))
                if not nbrs:
                    val = 0.0
                else:
                    diffs = [abs(self.graph.nodes[i]['phi'] - self.graph.nodes[j]['phi']) for j in nbrs]
                    # Wrap-around adjustment for angular distance
                    diffs = [min(d, 2 * math.pi - d) for d in diffs]
                    val = sum(diffs) / len(diffs)
                phi_grad[i] = val
        
        # Store results on graph
        for i, val in phi_grad.items():
            self.graph.nodes[i]['phi_grad'] = val
        return phi_grad

    def compute_phase_curvature(self) -> Dict[int, float]:
        """Compute K_φ(i) = φ_i - (1/deg(i)) Σ_{j∈N(i)} φ_j (neighbors undirected)."""
        if any('phi' not in self.graph.nodes[i] for i in self.graph.nodes()):
            self.compute_phase(store=True)
            
        # Use centralized cached function from physics.fields (CANONICAL)
        if HAS_CENTRALIZED_FIELDS:
            G = self._get_undirected_graph()
            k_phi = compute_phase_curvature(G)
        else:
            # Fallback implementation
            G = self._get_undirected_graph()
            k_phi = {}
            for i in G.nodes():
                nbrs = list(G.neighbors(i))
                if not nbrs:
                    val = 0.0
                else:
                    mean_phi = sum(self.graph.nodes[j]['phi'] for j in nbrs) / len(nbrs)
                    # Normalize curvature into principal range [-pi, pi) for interpretable magnitude
                    raw = self.graph.nodes[i]['phi'] - mean_phi
                    # Wrap to [-pi, pi)
                    while raw >= math.pi:
                        raw -= 2 * math.pi
                    while raw < -math.pi:
                        raw += 2 * math.pi
                    val = raw
                k_phi[i] = val
        
        # Store results on graph
        for i, val in k_phi.items():
            self.graph.nodes[i]['k_phi'] = val
        return k_phi

    def compute_kphi_safety(self, threshold: float = 3.0) -> Dict[str, float]:
        """Compute simple K_φ safety metric: fraction of nodes with |K_φ| ≥ threshold.

        Returns a dict with fields:
        - frac_abs_ge_threshold
        - count_abs_ge_threshold
        - total
        - threshold
        """
        if any('k_phi' not in self.graph.nodes[i] for i in self.graph.nodes()):
            self.compute_phase_curvature()
        vals = [abs(float(self.graph.nodes[i]['k_phi'])) for i in self.graph.nodes()]
        cnt = int(sum(v >= threshold for v in vals))
        total = len(vals) if vals else 1
        frac = cnt / total
        return {
            'frac_abs_ge_threshold': float(frac),
            'count_abs_ge_threshold': float(cnt),
            'total': float(total),
            'threshold': float(threshold),
        }

    def k_phi_multiscale_safety(
        self,
        *,
        distance_mode: str = 'arithmetic',
        max_r: Optional[int] = None,
        min_windows: int = 5,
        alpha_hint: float = 2.76,
    ) -> Dict[str, object]:
        """Estimate multiscale decay of K_φ variance and fit α in var ~ r^{-α}.

        For distance_mode='arithmetic', uses sliding windows of size r over the
        node index ordering (natural numbers). For 'topological', uses balls of
        graph distance ≤ r to compute per-node neighborhood means.

        Returns dict with:
        - r_list, var_list (per-scale variance of neighborhood-mean K_φ)
        - alpha_fit, R2_fit
        - alpha_hint, safe_flags based on AGENTS.md criteria
        """
        if any('k_phi' not in self.graph.nodes[i] for i in self.graph.nodes()):
            self.compute_phase_curvature()
        G = self._get_undirected_graph()
        nodes = sorted(G.nodes())
        kphi = np.array([float(self.graph.nodes[n]['k_phi']) for n in nodes], dtype=float)

        r_list: List[int] = []
        var_list: List[float] = []

        if distance_mode == 'arithmetic':
            # Use contiguous windows over natural index order
            N = len(nodes)
            if max_r is None:
                max_r = max(2, int(min(N // 3, max(5, math.sqrt(N)))))
            for r in range(1, max_r + 1):
                window_means = np.convolve(kphi, np.ones(r) / r, mode='valid')
                if window_means.size >= min_windows and np.var(window_means) > 0:
                    r_list.append(r)
                    var_list.append(float(np.var(window_means)))
        else:
            # Topological neighborhoods (balls of radius r)
            if not nx.is_connected(G):
                # Work with the largest connected component to avoid degenerate balls
                components = sorted(nx.connected_components(G), key=len, reverse=True)
                H = G.subgraph(components[0]).copy()
                nodes = sorted(H.nodes())
                G2 = H
            else:
                G2 = G
            # Precompute shortest paths
            spl = dict(nx.all_pairs_shortest_path_length(G2))
            diam = max(max(d.values()) for d in spl.values()) if spl else 1
            if max_r is None:
                max_r = max(2, min(diam, 10))
            # Build list
            for r in range(1, max_r + 1):
                means: List[float] = []
                for i in nodes:
                    neighborhood = [j for j, dist in spl[i].items() if dist <= r]
                    if neighborhood:
                        means.append(float(np.mean([self.graph.nodes[j]['k_phi'] for j in neighborhood])))
                if len(means) >= min_windows and np.var(means) > 0:
                    r_list.append(r)
                    var_list.append(float(np.var(means)))

        alpha_fit: Optional[float] = None
        R2_fit: Optional[float] = None
        intercept: Optional[float] = None
        slope: Optional[float] = None
        if len(var_list) >= 3 and all(v > 0 for v in var_list):
            xv = np.log(np.array(r_list, dtype=float))
            yv = np.log(np.array(var_list, dtype=float))
            coeffs = np.polyfit(xv, yv, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
            alpha_fit = float(-slope)
            y_pred = slope * xv + intercept
            ss_res = float(np.sum((yv - y_pred) ** 2))
            ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
            R2_fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # Safety flags per AGENTS.md
        safe_A = (alpha_fit is not None and alpha_fit > 0 and R2_fit is not None and R2_fit >= 0.5)
        # Check tolerance that observed aligns with alpha_hint approximately
        tol_ok = False
        if alpha_fit is not None:
            tol_ok = abs(alpha_fit - alpha_hint) <= max(0.5, 0.2 * alpha_hint)
        safe_flags = {
            'criterion_A_alpha_positive_R2_good': bool(safe_A),
            'criterion_B_alpha_close_to_hint': bool(tol_ok),
        }

        return {
            'r_list': r_list,
            'var_list': var_list,
            'alpha_fit': alpha_fit,
            'R2_fit': R2_fit,
            'slope_loglog': slope,
            'intercept_loglog': intercept,
            'alpha_hint': float(alpha_hint),
            'safe_flags': safe_flags,
            'distance_mode': distance_mode,
        }

    def compute_structural_potential(self, alpha: float = 2.0, distance_mode: str = "topological") -> Dict[int, float]:
        """
        Compute Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α.

        distance_mode:
        - 'topological': d(i,j) = shortest path length in undirected graph
        - 'arithmetic' : d(i,j) = |i - j|
        """
        G = self._get_undirected_graph()
        
        # Use centralized CANONICAL function when distance_mode is topological
        if HAS_CENTRALIZED_FIELDS and distance_mode == "topological":
            phi_s = centralized_phi_s(G, alpha=alpha)
        else:
            # Use arithmetic distance or fallback
            phi_s = self._cached_phi_s_helper(G, alpha=alpha, distance_mode=distance_mode)
            
        for i, v in phi_s.items():
            self.graph.nodes[i]['phi_s'] = v
        return phi_s

    def estimate_coherence_length(self, min_pairs: int = 5, distance_mode: str = "topological") -> Dict[str, object]:
        """
        Estimate coherence length ξ_C from spatial autocorrelation of local coherence c_i.

        c_i = 1 / (1 + |ΔNFR_i|)
        C(r) = ⟨c_i c_j⟩ where graph distance(i,j) = r (undirected)
        Fit: C(r) ≈ A * exp(-r / ξ_C) → ln C(r) = ln A - r / ξ_C
        """
        G = self._get_undirected_graph()
        # store local coherence (used by downstream analyses)
        for i in G.nodes():
            self.graph.nodes[i]['coherence_local'] = 1.0 / (1.0 + abs(self.graph.nodes[i]['DELTA_NFR']))
        
        # Use centralized CANONICAL function when distance_mode is topological
        if HAS_CENTRALIZED_FIELDS and distance_mode == "topological":
            # Centralized function returns scalar ξ_C, build compatible dict structure
            try:
                xi_c_value = centralized_xi_c(G, coherence_key="coherence_local")
                res = {
                    'r': [],
                    'C_r': [],
                    'xi_c': xi_c_value,
                    'fit_intercept': None,
                    'fit_slope': None,
                    'R2': None,
                }
            except Exception:
                # Fallback to custom implementation
                res = self._cached_xi_c_helper(G, min_pairs=min_pairs, distance_mode=distance_mode)
        else:
            # Use arithmetic distance or fallback
            res = self._cached_xi_c_helper(G, min_pairs=min_pairs, distance_mode=distance_mode)
            
        nodes = sorted(G.nodes())
        # enrich with safety baselines
        extra = {
            'system_diameter': (nx.diameter(G) if (distance_mode == 'topological' and nx.is_connected(G)) else (max(nodes) - min(nodes)) if distance_mode == 'arithmetic' else None),
            'mean_node_distance': (np.mean([d for _, d in nx.shortest_path_length(G)]) if (distance_mode == 'topological' and nx.is_connected(G)) else (np.mean(np.abs(np.subtract.outer(nodes, nodes))) if distance_mode == 'arithmetic' else None)),
        }
        res.update(extra)
        return res

    def compute_structural_fields(self, phase_method: str = "spectral") -> Dict[str, object]:
        """Convenience wrapper to compute all four structural telemetry fields."""
        phi = self.compute_phase(method=phase_method, store=True)
        grad = self.compute_phase_gradient()
        curv = self.compute_phase_curvature()
        phi_s = self.compute_structural_potential(alpha=2.0)
        xi = self.estimate_coherence_length()
        kphi_safety = self.compute_kphi_safety()
        kphi_multiscale = self.k_phi_multiscale_safety(distance_mode='arithmetic')
        return {
            'phi': phi,
            'phi_grad': grad,
            'k_phi': curv,
            'phi_s': phi_s,
            'xi_c': xi,
            'kphi_safety': kphi_safety,
            'kphi_multiscale': kphi_multiscale,
        }

    # ====================================================================
    # TNFR OPERATORS: COUPLING (UM) AND RESONANCE (RA)
    # ====================================================================

    def apply_coupling(self, delta_phi_max: float = math.pi / 2) -> Dict[Tuple[int, int], bool]:
        """Apply UM (Coupling): mark edges as coupled if phase compatible.

        Contract (U3): Only valid if |φ_i - φ_j| ≤ Δφ_max (wrapped on circle).
        Stores edge attribute 'coupled' = True/False on the undirected view.

        Returns a dict mapping undirected edge (min(i,j), max(i,j)) to boolean.
        """
        if any('phi' not in self.graph.nodes[i] for i in self.graph.nodes()):
            self.compute_phase(store=True)

        G = self._get_undirected_graph()
        coupled: Dict[Tuple[int, int], bool] = {}

        for u, v, data in G.edges(data=True):
            d = abs(self.graph.nodes[u]['phi'] - self.graph.nodes[v]['phi'])
            d = min(d, 2 * math.pi - d)  # circular distance
            is_ok = d <= float(delta_phi_max)
            data['coupled'] = bool(is_ok)
            key = (u, v) if u < v else (v, u)
            coupled[key] = is_ok

        # Telemetry: store fraction coupled
        total_edges = G.number_of_edges()
        frac = (sum(1 for v in coupled.values() if v) / total_edges) if total_edges > 0 else 0.0
        self.graph.graph['um_fraction_coupled'] = float(frac)
        self.graph.graph['um_delta_phi_max'] = float(delta_phi_max)
        return coupled

    def _neighbor_contrib(self, i: int, *, delta_phi_max: float) -> List[Tuple[int, float]]:
        """Helper: list of (j, weight) from neighbors j of i satisfying phase check."""
        G = self._get_undirected_graph()
        out: List[Tuple[int, float]] = []
        for j in G.neighbors(i):
            d = abs(self.graph.nodes[i].get('phi', 0.0) - self.graph.nodes[j].get('phi', 0.0))
            d = min(d, 2 * math.pi - d)
            if d <= delta_phi_max:
                # Use mean of parallel edges (gcd/divisibility) if exist in DiGraph
                w = 0.0
                if self.graph.has_edge(i, j):
                    w += float(self.graph[i][j].get('weight', 1.0))
                if self.graph.has_edge(j, i):
                    w += float(self.graph[j][i].get('weight', 1.0))
                if w == 0.0:
                    w = 1.0
                out.append((j, w))
        return out

    def resonance_step(
        self,
        activation: Dict[int, float],
        *,
        gain: float = 1.0,
        decay: float = 0.0,
        delta_phi_max: float = math.pi / 2,
        normalize: bool = True,
    ) -> Dict[int, float]:
        """Apply one RA (Resonance) step on an activation field.

        Physics (RA): Propagates patterns coherently through coupled links, without
        altering identity (keeps activation as a separate field; does not mutate EPI).
        Contract: Requires phase verification (U3) via delta_phi_max.
        """
        G = self._get_undirected_graph()
        # Ensure phases available
        if any('phi' not in self.graph.nodes[n] for n in G.nodes()):
            self.compute_phase(store=True)

        new_act: Dict[int, float] = {}
        for i in G.nodes():
            acc = 0.0
            contribs = self._neighbor_contrib(i, delta_phi_max=delta_phi_max)
            total_w = sum(w for _, w in contribs) or 1.0
            for j, w in contribs:
                acc += (w / total_w) * activation.get(j, 0.0)
            # Gain/decay dynamics
            val = (1 - decay) * activation.get(i, 0.0) + gain * acc
            new_act[i] = max(0.0, float(val))

        if normalize and new_act:
            m = max(new_act.values())
            if m > 0:
                for k in new_act:
                    new_act[k] /= m
        return new_act

    def resonance_from_primes(
        self,
        steps: int = 5,
        *,
        init_value: float = 1.0,
        gain: float = 1.0,
        decay: float = 0.1,
        delta_phi_max: float = math.pi / 2,
        normalize: bool = True,
        seed: Optional[int] = None,
        jitter: bool = True,
    ) -> List[Dict[int, float]]:
        """Seed activation on primes and run RA propagation.

        Returns a list of activation dicts for steps [0..steps].
        Step 0 is the seed activation (possibly jittered for seed fairness).
        """
        if seed is not None:
            # Controlled determinism for activation jitter
            import random
            random.seed(seed)
            np.random.seed(seed)
        primes = [
            n for n in self.graph.nodes()
            if self.graph.nodes[n].get('is_prime', False)
        ]
        act: Dict[int, float] = {
            n: (init_value if n in primes else 0.0)
            for n in self.graph.nodes()
        }
        if normalize:
            # normalizing initial seed
            m0 = max(act.values()) if act else 1.0
            if m0 > 0:
                for k in act:
                    act[k] /= m0
        if jitter and primes:
            # Introduce infinitesimal jitter to distinguish seeds if needed
            for p in primes:
                act[p] *= (1.0 + np.random.uniform(-1e-6, 1e-6))

        history: List[Dict[int, float]] = [dict(act)]
        # Apply UM for bookkeeping/telemetry
        self.apply_coupling(delta_phi_max=delta_phi_max)
        for _ in range(steps):
            act = self.resonance_step(
                act,
                gain=gain,
                decay=decay,
                delta_phi_max=delta_phi_max,
                normalize=normalize,
            )
            history.append(dict(act))
        # Telemetry metrics
        self.graph.graph['ra_steps'] = int(steps)
        self.graph.graph['ra_gain'] = float(gain)
        self.graph.graph['ra_decay'] = float(decay)
        self.graph.graph['ra_delta_phi_max'] = float(delta_phi_max)
        return history

    def resonance_metrics(
        self, activation: Dict[int, float]
    ) -> Dict[str, float]:
        """Compute simple metrics on an activation field for analysis."""
        values = np.array(list(activation.values()), dtype=float)
        mean_act = float(np.mean(values)) if values.size else 0.0
        frac_high = float(np.mean(values >= 0.5)) if values.size else 0.0
        # Correlation with prime indicator
        y = np.array([
            1.0 if self.graph.nodes[n].get('is_prime', False) else 0.0
            for n in sorted(self.graph.nodes())
        ], dtype=float)
        x = np.array([
            activation.get(n, 0.0)
            for n in sorted(self.graph.nodes())
        ], dtype=float)
        if x.size and y.size:
            xm, ym = x - x.mean(), y - y.mean()
            denom = float(np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum()))
            corr = float((xm * ym).sum() / denom) if denom > 0 else 0.0
        else:
            corr = 0.0
        return {
            'mean_activation': mean_act,
            'fraction_ge_0_5': frac_high,
            'corr_with_primes': corr,
        }

    # ====================================================================
    # TELEMETRY EXPORT FUNCTIONS (JSONL / CSV)
    # ====================================================================

    def export_prime_certificates(
        self,
        path: str,
        *,
        delta_nfr_threshold: float = 0.2,
        fmt: str = "jsonl",
        include_components: bool = True,
    ) -> int:
        """Export prime certificates for all numbers up to max_number.

        Parameters
        ----------
        path : str
            Output file path.
        delta_nfr_threshold : float, default 0.2
            Threshold used for candidate detection (telemetry context).
        fmt : {"jsonl","csv"}, default "jsonl"
            Output format.
        include_components : bool, default True
            Include component breakdown (zeta/eta/theta contributions).

        Returns
        -------
        int
            Number of certificates exported.
        """
        import json
        import csv
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        candidates = self.detect_prime_candidates(
            delta_nfr_threshold=delta_nfr_threshold
        )
        count = 0
        if fmt.lower() == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for n, _delta in candidates:
                    cert = self.get_prime_certificate(n)
                    data = cert.as_dict()
                    if not include_components:
                        data.pop("components", None)
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    count += 1
        elif fmt.lower() == "csv":
            # Determine fields
            sample_cert = (
                self.get_prime_certificate(candidates[0][0]).as_dict()
                if candidates else {}
            )
            if not include_components:
                sample_cert.pop("components", None)
            fields = list(sample_cert.keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for n, _delta in candidates:
                    cert = self.get_prime_certificate(n).as_dict()
                    if not include_components:
                        cert.pop("components", None)
                    writer.writerow(cert)
                    count += 1
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(
            f"Exported {count} prime certificates to {path} (fmt={fmt})"
        )
        return count

    def export_structural_fields(
        self,
        path: str,
        *,
        phase_method: str = "logn",
        fmt: str = "jsonl",
    ) -> int:
        """Export structural field telemetry (φ, |∇φ|, K_φ, Φ_s, ξ_C).

        The coherence length ξ_C is exported as its summary entries.
        Returns number of nodes exported.
        """
        import json
        import csv
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = self.compute_structural_fields(phase_method=phase_method)
        phi = fields['phi']
        grad = fields['phi_grad']
        kphi = fields['k_phi']
        phi_s = fields['phi_s']
        xi = fields['xi_c']  # dict with summary stats
        nodes = sorted(self.graph.nodes())
        count = 0
        if fmt.lower() == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for n in nodes:
                    row = {
                        'n': n,
                        'phi': float(phi.get(n, 0.0)),
                        'phi_grad': float(grad.get(n, 0.0)),
                        'k_phi': float(kphi.get(n, 0.0)),
                        'phi_s': float(phi_s.get(n, 0.0)),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            # Append coherence length summary as a separate JSON object
            with open(path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({'xi_c_summary': xi}, ensure_ascii=False) + "\n"
                )
        elif fmt.lower() == "csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["n", "phi", "phi_grad", "k_phi", "phi_s"])
                for n in nodes:
                    writer.writerow([
                        n,
                        float(phi.get(n, 0.0)),
                        float(grad.get(n, 0.0)),
                        float(kphi.get(n, 0.0)),
                        float(phi_s.get(n, 0.0)),
                    ])
                    count += 1
                # Write ξ_C summary below as key,value pairs
                writer.writerow([])
                writer.writerow(["xi_c_summary_key", "xi_c_summary_value"])
                for k, v in xi.items():
                    writer.writerow([k, v])
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(
            "Exported structural fields for %d nodes to %s (fmt=%s)" % (
                count,
                path,
                fmt,
            )
        )
        return count


# ============================================================================
# TESTING AND VALIDATION FUNCTIONS
# ============================================================================

def run_basic_validation(max_number: int = 50) -> None:
    """Run basic validation of TNFR prime detection."""
    logger.info("=" * 60)
    logger.info("TNFR ARITHMETIC NETWORK: Prime Detection Validation")
    logger.info("=" * 60)
    
    # Create network
    logger.info(f"Creating arithmetic TNFR network (n ≤ {max_number})...")
    network = ArithmeticTNFRNetwork(max_number)
    
    # Summary statistics
    stats = network.summary_statistics()
    logger.info("Network Statistics:")
    logger.info(f"  Total numbers: {stats['total_numbers']}")
    logger.info(f"  Known primes: {stats['prime_count']}")
    logger.info(f"  Prime ratio: {stats['prime_ratio']:.3f}")
    logger.info(f"  Prime mean ΔNFR: {stats['prime_mean_DELTA_NFR']:.6f}")
    logger.info(
        f"  Composite mean ΔNFR: {stats['composite_mean_DELTA_NFR']:.6f}"
    )
    logger.info(f"  ΔNFR separation: {stats['DELTA_NFR_separation']:.6f}")
    
    # Test prime detection
    logger.info("Testing prime detection...")
    validation = network.validate_prime_detection(delta_nfr_threshold=0.1)
    logger.info(f"  Precision: {validation['precision']:.3f}")
    logger.info(f"  Recall: {validation['recall']:.3f}")
    logger.info(f"  F1-score: {validation['f1_score']:.3f}")
    
    if validation['false_alarms']:
        logger.info(f"  False alarms: {validation['false_alarms']}")
    if validation['missed_primes']:
        logger.info(f"  Missed primes: {validation['missed_primes']}")
    
    # Show first few primes
    logger.info("First 10 primes with TNFR properties:")
    primes = [n for n in range(2, max_number + 1) if network._is_prime(n)][:10]
    for p in primes:
        props = network.get_tnfr_properties(p)
        logger.info(
            "  %2d: EPI=%.3f, νf=%.3f, ΔNFR=%.6f" % (
                p,
                props['EPI'],
                props['nu_f'],
                props['DELTA_NFR'],
            )
        )


if __name__ == "__main__":
    # Run basic validation
    run_basic_validation(max_number=100)
