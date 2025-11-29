"""🌊 **Simple TNFR SDK** - Maximum Power, Minimum Complexity ⭐

The new simplified TNFR API designed for 90% of use cases.
Three classes, endless possibilities.

**DESIGN PRINCIPLE**: 
- **Intuitive**: Natural method names that read like English
- **Chainable**: Fluent interface for rapid prototyping  
- **Complete**: Full TNFR physics under the hood
- **Fast**: Optimized for common patterns

**USAGE EXAMPLES**:

>>> # Instant network creation
>>> net = TNFR.create(20)  # 20 nodes

>>> # Chain operations
>>> results = TNFR.create(10).ring().evolve(5).results()

>>> # Auto-optimization
>>> optimized = TNFR.create(15).random(0.3).auto_optimize()

>>> # Templates
>>> molecular = TNFR.template('molecule')

>>> # Analysis
>>> coherence = net.coherence()
>>> summary = net.summary()
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import networkx as nx
import numpy as np

# TNFR core imports
from ..structural import create_nfr, run_sequence
from ..metrics.coherence import compute_coherence
from ..metrics.sense_index import compute_Si

try:
    from ..dynamics.self_optimizing_engine import TNFRSelfOptimizingEngine
    _HAS_OPTIMIZATION = True
except ImportError:
    _HAS_OPTIMIZATION = False


@dataclass
class Results:
    """🏆 **Lightweight TNFR Results** - Essential Metrics Only.
    
    Contains the most important metrics from TNFR analysis.
    Designed for quick inspection and comparison.
    """
    coherence: float
    sense_index: float
    nodes: int
    edges: int
    density: float
    avg_phase: float
    
    def summary(self) -> str:
        """📊 One-line summary of results."""
        # Convert numpy arrays to float for formatting
        coherence = float(self.coherence) if hasattr(self.coherence, 'item') else self.coherence
        sense_index = float(self.sense_index) if hasattr(self.sense_index, 'item') else self.sense_index
        density = float(self.density) if hasattr(self.density, 'item') else self.density
        
        return (f"C={coherence:.3f}, Si={sense_index:.3f}, "
                f"N={self.nodes}, E={self.edges}, ρ={density:.3f}")
    
    def is_coherent(self) -> bool:
        """✅ Quick coherence check (C > 0.7)."""
        return self.coherence > 0.7
    
    def is_stable(self) -> bool:
        """🔒 Quick stability check (Si > 0.8)."""
        return self.sense_index > 0.8


class Network:
    """🕸️ **Core TNFR Network** - Essential Operations Only.
    
    Simplified interface to TNFR networks with most common operations.
    Focus on the 90% of functionality users actually need.
    """
    
    def __init__(self, graph: nx.Graph, name: str = "network"):
        """Initialize with a NetworkX graph."""
        self.G = graph
        self.name = name
    
    # === TOPOLOGY BUILDERS ===
    
    def ring(self) -> Network:
        """🔄 Connect nodes in a ring topology."""
        nodes = list(self.G.nodes())
        edges = [(nodes[i], nodes[(i+1) % len(nodes)]) for i in range(len(nodes))]
        self.G.add_edges_from(edges)
        return self
    
    def complete(self) -> Network:
        """🌐 Connect all nodes to all other nodes."""
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                self.G.add_edge(u, v)
        return self
    
    def random(self, probability: float = 0.3) -> Network:
        """🎲 Add random connections with given probability."""
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if np.random.random() < probability:
                    self.G.add_edge(u, v)
        return self
    
    def star(self, center: Optional[int] = None) -> Network:
        """⭐ Create star topology with optional center node."""
        nodes = list(self.G.nodes())
        if center is None:
            center = nodes[0]
        
        for node in nodes:
            if node != center:
                self.G.add_edge(center, node)
        return self
    
    # === EVOLUTION ===
    
    def evolve(self, steps: int = 5, sequence: str = "basic_activation") -> Network:
        """🧬 Evolve network using TNFR dynamics."""
        # Ensure nodes have TNFR properties
        if not all('EPI' in self.G.nodes[n] for n in self.G.nodes()):
            create_nfr(self.G)
        
        # Simple evolution: Apply coherence improvement to all nodes
        from ..operators.definitions import Coherence, Resonance
        
        for step in range(steps):
            for node in self.G.nodes():
                try:
                    # Simple sequence: Coherence -> Resonance
                    ops = [Coherence(), Resonance()]
                    run_sequence(self.G, node, ops)
                except Exception:
                    # If sequence fails, just continue
                    continue
        
        return self
    
    def auto_optimize(self) -> Network:
        """🤖 Auto-optimize using self-optimizing engine."""
        if not _HAS_OPTIMIZATION:
            print("⚠️  Auto-optimization not available - using basic evolution")
            return self.evolve(3, "stabilization")
        
        if not all('EPI' in self.G.nodes[n] for n in self.G.nodes()):
            create_nfr(self.G)
        
        engine = TNFRSelfOptimizingEngine(self.G)
        
        # Try to optimize a few nodes
        for node in list(self.G.nodes())[:min(5, len(self.G.nodes()))]:
            try:
                engine.step(node)
            except Exception:
                continue  # Skip if optimization fails
        
        return self
    
    # === METRICS ===
    
    def coherence(self) -> float:
        """📏 Current network coherence [0,1]."""
        result = compute_coherence(self.G)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)
    
    def sense_index(self) -> float:
        """🎯 Current sense index [0,1+]."""
        result = compute_Si(self.G)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)
    
    def density(self) -> float:
        """🔗 Network density [0,1]."""
        n = len(self.G.nodes())
        if n < 2:
            return 0.0
        return 2 * len(self.G.edges()) / (n * (n - 1))
    
    def avg_phase(self) -> float:
        """📐 Average node phase [0, 2π]."""
        if not self.G.nodes():
            return 0.0
        phases = [self.G.nodes[n].get('phase', 0) for n in self.G.nodes()]
        result = np.mean(phases)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)
    
    def results(self) -> Results:
        """🏆 Get comprehensive results."""
        return Results(
            coherence=self.coherence(),
            sense_index=self.sense_index(), 
            nodes=len(self.G.nodes()),
            edges=len(self.G.edges()),
            density=self.density(),
            avg_phase=self.avg_phase()
        )
    
    def summary(self) -> str:
        """📋 Quick network summary."""
        return self.results().summary()
    
    # === ANALYSIS ===
    
    def info(self) -> Dict[str, Any]:
        """ℹ️  Detailed network information."""
        return {
            'name': self.name,
            'nodes': len(self.G.nodes()),
            'edges': len(self.G.edges()),
            'density': self.density(),
            'coherence': self.coherence(),
            'sense_index': self.sense_index(),
            'avg_phase': self.avg_phase(),
            'is_connected': nx.is_connected(self.G),
            'has_tnfr_props': all('EPI' in self.G.nodes[n] for n in self.G.nodes())
        }


class TNFR:
    """🌊 **Static Factory for Instant TNFR Networks** ⭐
    
    Main entry point for the simplified TNFR SDK.
    All methods are static for maximum convenience.
    
    **PHILOSOPHY**: Start creating networks immediately with zero boilerplate.
    """
    
    @staticmethod
    def create(num_nodes: int, name: str = "network") -> Network:
        """🏗️ Create empty TNFR network with specified nodes.
        
        Args:
            num_nodes: Number of nodes to create
            name: Optional network name
            
        Returns:
            Network ready for topology and evolution
            
        Example:
            >>> net = TNFR.create(10)  # 10 isolated nodes
        """
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Initialize with TNFR properties
        create_nfr(G)
        
        return Network(G, name)
    
    @staticmethod
    def template(template_name: str) -> Network:
        """📋 Create network from pre-configured template.
        
        Available templates:
        - 'small': 5 nodes, ring topology
        - 'medium': 15 nodes, small-world topology  
        - 'large': 50 nodes, random topology
        - 'molecule': 8 nodes, molecular-like structure
        - 'star': 10 nodes, star topology
        - 'complete': 6 nodes, complete graph
        
        Args:
            template_name: Template to use
            
        Returns:
            Pre-configured network ready to use
            
        Example:
            >>> mol = TNFR.template('molecule')
        """
        templates = {
            'small': lambda: TNFR.create(5).ring(),
            'medium': lambda: TNFR.create(15).ring().random(0.1),  # Small-world-like
            'large': lambda: TNFR.create(50).random(0.08),
            'molecule': lambda: TNFR.create(8).ring().random(0.2),
            'star': lambda: TNFR.create(10).star(),
            'complete': lambda: TNFR.create(6).complete()
        }
        
        if template_name not in templates:
            available = ', '.join(templates.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
        
        return templates[template_name]()
    
    @staticmethod
    def compare(*networks: Network) -> Dict[str, Any]:
        """⚖️ Compare multiple networks side by side.
        
        Args:
            *networks: Networks to compare
            
        Returns:
            Comparison results with rankings
            
        Example:
            >>> comparison = TNFR.compare(net1, net2, net3)
            >>> print(comparison['ranking'])
        """
        if not networks:
            return {}
        
        results = []
        for i, net in enumerate(networks):
            result = net.results()
            results.append({
                'name': net.name,
                'index': i,
                'coherence': result.coherence,
                'sense_index': result.sense_index,
                'nodes': result.nodes,
                'edges': result.edges,
                'density': result.density
            })
        
        # Rank by coherence
        ranking = sorted(results, key=lambda x: x['coherence'], reverse=True)
        
        return {
            'results': results,
            'ranking': ranking,
            'best': ranking[0] if ranking else None,
            'worst': ranking[-1] if ranking else None,
            'count': len(networks)
        }


# === CONVENIENT ALIASES ===

# Short aliases for power users
T = TNFR  # Even shorter: T.create(10).ring()
Net = Network  # Type alias

# Export main API
__all__ = ['TNFR', 'Network', 'Results', 'T', 'Net']