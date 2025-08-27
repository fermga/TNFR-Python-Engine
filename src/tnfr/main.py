from __future__ import annotations
import argparse, sys
import networkx as nx
from . import preparar_red, run, __version__

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="tnfr",
        description="TNFR canónica — demo CLI (red Erdős–Rényi + dinámica glífica)",
    )
    p.add_argument("--version", action="store_true", help="muestra versión y sale")
    p.add_argument("--n", type=int, default=30, help="nodos (Erdős–Rényi)")
    p.add_argument("--p", type=float, default=0.15, help="probabilidad de arista (Erdős–Rényi)")
    p.add_argument("--steps", type=int, default=100, help="pasos a simular")
    p.add_argument("--observer", action="store_true", help="adjunta observador estándar")
    args = p.parse_args(argv)

    if args.version:
        print(__version__)
        return

    G = nx.erdos_renyi_graph(args.n, args.p)
    preparar_red(G, ATTACH_STD_OBSERVER=bool(args.observer))
    run(G, args.steps)

    h = G.graph.get("history", {})
    C = h.get("C_steps", [])[-1] if h.get("C_steps") else None
    stab = h.get("stable_frac", [])[-1] if h.get("stable_frac") else None
    R = h.get("kuramoto_R", [])[-1] if h.get("kuramoto_R") else None

    print("TNFR terminado:")
    if C is not None: print(f" C(t) ~ {C:.3f}")
    if stab is not None: print(f" estable ~ {stab:.3f}")
    if R is not None: print(f" R (Kuramoto) ~ {R:.3f}")

if __name__ == "__main__":
    main(sys.argv[1:])
