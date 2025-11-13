# docs/research/tnfr_research_utils.py
"""
Reusable utilities for TNFR research notebooks, providing canonical implementations
for structural field metrics and grammar-compliant operator sequences (motifs).
"""
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

# --- Canonical Field Metrics ---

# Try to import canonical field functions from the core library
_fields_api: Dict[str, Callable] = {}
try:
    from src.tnfr.physics import fields as _fields

    # Structural potential (Φ_s) - check for multiple common names
    phi_s_fn: Callable = getattr(_fields, 'compute_structural_potential', None) or \
        getattr(_fields, 'structural_potential', None)
    # Phase gradient (|∇φ|)
    grad_phi_fn: Callable = getattr(_fields, 'compute_phase_gradient', None)
    # Phase curvature (K_φ)
    k_phi_fn: Callable = getattr(_fields, 'compute_phase_curvature', None)
    # Coherence length (ξ_C)
    xi_c_fn: Callable = getattr(_fields, 'estimate_coherence_length', None)

    _fields_api = {
        'phi_s': phi_s_fn,
        'grad': grad_phi_fn,
        'kphi': k_phi_fn,
        'xi_c': xi_c_fn,
    }
    print("✅ Canonical fields API loaded successfully.")
except ImportError:
    print("⚠️ Could not import from 'src.tnfr.physics.fields'. Using fallback estimators.")
    _fields_api = {}
except Exception as e:
    print(f"🚨 Error loading canonical fields API: {e}")
    _fields_api = {}


# --- Fallback Estimators ---
# These are used ONLY if the canonical API from src.tnfr.physics.fields is unavailable.


def _fallback_structural_potential(G) -> float:
    """Simple inverse-distance-like proxy using degrees as distance surrogate."""
    n = len(G.nodes)
    if n <= 1:
        return 0.0
    degs = [G.degree(v) for v in G.nodes]
    s = sum(degs)
    return float(s) / max(1, n * (n - 1))


def _fallback_phase_gradient(G) -> float:
    """Mean absolute phase difference over edges if 'theta' or 'phase' present."""
    def get_phi(v):
        d = G.nodes[v]
        return d.get('theta', d.get('phase', 0.0))

    diffs = [abs(get_phi(u) - get_phi(v)) for u, v in G.edges]
    return float(sum(diffs)) / len(diffs) if diffs else 0.0


def _fallback_phase_curvature(G) -> float:
    """Max absolute curvature proxy: node phase minus neighbor mean phase."""
    def get_phi(v):
        d = G.nodes[v]
        return d.get('theta', d.get('phase', 0.0))

    vals = []
    for i in G.nodes:
        nbrs = list(G.neighbors(i))
        if not nbrs:
            continue
        mean_n = sum(get_phi(j) for j in nbrs) / len(nbrs)
        vals.append(abs(get_phi(i) - mean_n))
    return max(vals) if vals else 0.0


def _fallback_xi_c(G) -> float:
    """Placeholder: returns 0.0 as a non-canonical fallback."""
    return 0.0


# --- Unified Tetrad Computation ---


# Select active functions: canonical if available, otherwise fallbacks
_phi_s = _fields_api.get('phi_s') or _fallback_structural_potential
_grad = _fields_api.get('grad') or _fallback_phase_gradient
_kphi = _fields_api.get('kphi') or _fallback_phase_curvature
_xi_c = _fields_api.get('xi_c') or _fallback_xi_c


def _graph_fingerprint(G) -> Tuple[int, int, float]:
    """Coarse, fast fingerprint for caching based on graph structure and phase."""
    n = len(G.nodes)
    m = len(G.edges)
    sphi = sum(d.get('theta', d.get('phase', 0.0)) for _, d in G.nodes(data=True))
    return (n, m, round(sphi, 6))


@lru_cache(maxsize=256)
def _cached_compute_tetrad(fp: Tuple[int, int, float], G) -> Dict[str, float]:
    """LRU-cached computation of the structural field tetrad."""
    # This function is wrapped to provide caching; direct calls go to compute_tetrad.
    # The fingerprint `fp` is used by lru_cache, while `G` is passed for computation.
    results = {}
    try:
        results['phi_s'] = float(_phi_s(G))
    except Exception:
        results['phi_s'] = float(_fallback_structural_potential(G))
    try:
        results['grad'] = float(_grad(G))
    except Exception:
        results['grad'] = float(_fallback_phase_gradient(G))
    try:
        results['kphi'] = float(_kphi(G))
    except Exception:
        results['kphi'] = float(_fallback_phase_curvature(G))
    try:
        results['xi_c'] = float(_xi_c(G))
    except Exception:
        results['xi_c'] = float(_fallback_xi_c(G))
    return results


def compute_tetrad(G, use_cache: bool = True) -> Dict[str, float]:
    """
    Computes the structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) for a given graph.

    Uses canonical implementations from `src.tnfr.physics.fields` if available,
    with safe fallbacks. Caching can be disabled for fresh computations.
    """
    if use_cache:
        fp = _graph_fingerprint(G)
        return _cached_compute_tetrad(fp, G)
    return _cached_compute_tetrad.__wrapped__(None, G)


# --- Operator Sequence Motifs ---


def get_motifs(set_name: str = 'generator_first') -> Dict[str, List[str]]:
    """
    Returns a dictionary of specified grammar-compliant operator sequences (motifs).

    Available sets:
    - 'generator_first': U1a compliant, starts with 'emission'. Recommended.
    - 'stabilize_only': Minimalist stabilization sequence.
    - 'gentle_bond': U4b compliant mutation sequence.
    - 'stabilize_sandwich': Repeated dissonance/coherence cycles.
    """
    motifs = {
        'generator_first': {
            "stabilize_only_gen": ["emission", "coherence", "silence"],
            "gentle_bond_gen": ["emission", "coherence", "dissonance", "coherence", "mutation", "coherence", "silence"],
            "stabilize_sandwich_gen": ["emission", "coherence", "dissonance", "coherence", "dissonance", "coherence", "silence"],
        },
        'stabilize_only': {
            "stabilize_only": ["coherence", "silence"],
        },
        'gentle_bond': {
            "gentle_bond": ["coherence", "dissonance", "coherence", "mutation", "coherence", "silence"],
        },
        'stabilize_sandwich': {
            "stabilize_sandwich": ["coherence", "dissonance", "coherence", "dissonance", "coherence", "silence"],
        }
    }
    return motifs.get(set_name, motifs['generator_first'])
