"""Type stubs for tnfr.extensions.medical.health_analyzers"""

from typing import Any, Dict, List
import networkx as nx

class TherapeuticHealthAnalyzer:
    def analyze_therapeutic_health(
        self, G: nx.Graph, sequence: List[str], **kwargs: Any
    ) -> Dict[str, float]: ...
