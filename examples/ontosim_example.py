"""Pequeño ejemplo de uso de ``tnfr.ontosim``.

Genera un grafo Erdős–Rényi y ejecuta 100 pasos de la simulación usando los
valores por defecto.
"""

import networkx as nx

from tnfr.ontosim import preparar_red, run


def main() -> None:
    G = nx.erdos_renyi_graph(30, 0.15)
    preparar_red(G)
    run(G, 100)
    # print("C(t) muestras:", G.graph["history"]["C_steps"][-5:])  # usado solo para pruebas


if __name__ == "__main__":  # pragma: no cover - ejemplo manual
    main()
