"""Benchmark Utilities for TNFR Research
==========================================

Shared utilities for creating valid TNFR graphs, generating grammar-compliant
operator sequences, and running experiments.

Status: RESEARCH - Support infrastructure for validation experiments
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Any

import networkx as nx

# Import TNFR core
import sys
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.config import (
    EPI_PRIMARY,
    VF_PRIMARY,
    DNFR_PRIMARY,
    THETA_PRIMARY,
    inject_defaults,
)
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Resonance, Coupling,
    SelfOrganization, Mutation, Silence, Expansion, Contraction,
    Transition, Recursivity,
)


def create_tnfr_topology(
    topology: str,
    n_nodes: int,
    seed: int
) -> nx.Graph:
    """Create a network topology with proper TNFR initialization.
    
    Parameters
    ----------
    topology : str
        One of: 'ring', 'scale_free', 'ws', 'tree', 'grid'
    n_nodes : int
        Number of nodes
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    nx.Graph
        Graph with TNFR defaults injected but nodes not yet initialized
    """
    random.seed(seed)
    
    if topology == 'ring':
        G = nx.cycle_graph(n_nodes)
    elif topology == 'scale_free':
        G = nx.scale_free_graph(n_nodes, seed=seed).to_undirected()
    elif topology == 'ws':  # small-world
        k = min(4, n_nodes - 1) if n_nodes > 1 else 0
        G = nx.watts_strogatz_graph(n_nodes, k=k, p=0.3, seed=seed)
    elif topology == 'tree':
        if n_nodes <= 1:
            G = nx.Graph()
            G.add_node(0)
        else:
            height = max(1, int(math.log2(n_nodes)))
            G = nx.balanced_tree(r=2, h=height)
            if G.number_of_nodes() > n_nodes:
                nodes_to_remove = list(G.nodes)[n_nodes:]
                G.remove_nodes_from(nodes_to_remove)
    elif topology == 'grid':
        if n_nodes <= 1:
            G = nx.Graph()
            G.add_node(0)
        else:
            side = max(2, int(math.sqrt(n_nodes)))
            G = nx.grid_2d_graph(side, side)
            # Convert to integer node labels
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            if G.number_of_nodes() > n_nodes:
                nodes_to_remove = list(G.nodes)[n_nodes:]
                G.remove_nodes_from(nodes_to_remove)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Inject TNFR defaults into graph
    inject_defaults(G)
    
    return G


def initialize_tnfr_nodes(
    G: nx.Graph,
    nu_f: float = 1.0,
    epi_range: tuple[float, float] = (0.2, 0.8),  # FIXED: Within [-1, 1]
    seed: int = 42,
) -> None:
    """Initialize node attributes with proper TNFR primary keys.
    
    Parameters
    ----------
    G : nx.Graph
        Graph to initialize (must have inject_defaults already called)
    nu_f : float
        Structural frequency (νf) for all nodes
    epi_range : tuple[float, float]
        (min, max) for random EPI initialization (must be in [-1, 1])
    seed : int
        Random seed
    """
    random.seed(seed)
    
    for node in G.nodes:
        G.nodes[node][EPI_PRIMARY] = random.uniform(*epi_range)
        G.nodes[node][VF_PRIMARY] = nu_f
        G.nodes[node][THETA_PRIMARY] = random.uniform(0.0, 2 * math.pi)
        G.nodes[node][DNFR_PRIMARY] = random.uniform(0.01, 0.05)
        # NOTE: Do NOT add legacy aliases ('phase', 'delta_nfr')
        # The fields.py module will find the correct aliases automatically
        # Adding redundant aliases causes desync issues


def generate_grammar_valid_sequence(
    sequence_type: str,
    intensity: float = 1.0,
) -> List[Any]:
    """Generate operator sequences that comply with TNFR grammar (U1-U4).
    
    All sequences:
    - Start with generator (U1a): Emission, Transition, or Recursivity
    - End with closure (U1b): Silence, Transition, Recursivity, or Dissonance
    - Include stabilizers after destabilizers (U2)
    - Respect canonical flow compatibility matrix
    
    Parameters
    ----------
    sequence_type : str
        'RA_dominated' (resonance-heavy), 'OZ_heavy' (dissonance-heavy),
        'balanced' (mixed)
    intensity : float
        Multiplier for sequence length (1.0 = baseline)
        
    Returns
    -------
    List
        List of operator instances forming a valid sequence
    """
    base_length = max(3, int(intensity * 5))
    
    if sequence_type == 'RA_dominated':
        # Resonance/coupling focused sequence
        # Pattern: AL → IL → (RA → IL)* → SHA
        seq = [Emission(), Coherence()]
        n_resonance = max(1, int(base_length * 0.5))
        for _ in range(n_resonance):
            seq.extend([Resonance(), Coherence()])
        seq.append(Silence())
        
    elif sequence_type == 'OZ_heavy':
        # Dissonance-heavy sequence with required stabilizers
        # Pattern: AL → (OZ → IL)* → (VAL → THOL) → SHA
        seq = [Emission()]
        n_dissonance = max(1, int(base_length * 0.4))
        for _ in range(n_dissonance):
            seq.extend([Dissonance(), Coherence()])
        # Add expansion with self-organization (U2: destabilizer needs stabilizer)
        seq.extend([Dissonance(), Expansion(), SelfOrganization(), Coherence()])
        seq.append(Silence())
        
    elif sequence_type == 'balanced':
        # Mix of operators maintaining grammar compliance
        # Pattern: AL → IL → (OZ → IL → RA → IL)* → SHA
        seq = [Emission(), Coherence()]
        n_cycles = max(1, int(base_length * 0.3))
        for _ in range(n_cycles):
            seq.extend([Dissonance(), Coherence(), Resonance(), Coherence()])
        seq.append(Silence())
        
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")
    
    return seq


# Validation: Ensure sequences comply with grammar
def validate_sequence_grammar(sequence: List[Any]) -> tuple[bool, str]:
    """Validate that a sequence complies with TNFR grammar rules.
    
    Checks:
    - U1a: Starts with generator (AL, NAV, REMESH)
    - U1b: Ends with closure (SHA, NAV, REMESH, OZ)
    - U2: Destabilizers (OZ, VAL, ZHIR) followed by stabilizers (IL, THOL)
    
    Parameters
    ----------
    sequence : List
        List of operator instances
        
    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    if not sequence:
        return False, "Empty sequence"
    
    # Map operators to their names
    op_names = [op.__class__.__name__ for op in sequence]
    
    # U1a: Generator check
    generators = {'Emission', 'Transition', 'Recursivity'}
    if op_names[0] not in generators:
        return False, f"Must start with generator, got {op_names[0]}"
    
    # U1b: Closure check
    closures = {'Silence', 'Transition', 'Recursivity', 'Dissonance'}
    if op_names[-1] not in closures:
        return False, f"Must end with closure, got {op_names[-1]}"
    
    # U2: Destabilizer → Stabilizer check
    destabilizers = {'Dissonance', 'Expansion', 'Mutation'}
    stabilizers = {'Coherence', 'SelfOrganization'}
    
    for i, op_name in enumerate(op_names):
        if op_name in destabilizers:
            # Look ahead for stabilizer within next 3 operators
            found_stabilizer = False
            for j in range(i + 1, min(i + 4, len(op_names))):
                if op_names[j] in stabilizers:
                    found_stabilizer = True
                    break
            if not found_stabilizer:
                return False, f"Destabilizer {op_name} at position {i} needs stabilizer within 3 ops"
    
    return True, "Valid"


__all__ = [
    'create_tnfr_topology',
    'initialize_tnfr_nodes',
    'generate_grammar_valid_sequence',
    'validate_sequence_grammar',
]
