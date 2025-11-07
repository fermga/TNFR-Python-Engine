"""Type stubs for tnfr.extensions.business.health_analyzers"""

from typing import Any, Dict, List
import networkx as nx

class ProcessHealthAnalyzer:
    def analyze_process_health(
        self, G: nx.Graph, sequence: List[str], **kwargs: Any
    ) -> Dict[str, float]: ...
