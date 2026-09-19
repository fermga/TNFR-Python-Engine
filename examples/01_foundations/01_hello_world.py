"""Create a small TNFR network and observe one supplied operator word.

The public SDK owns initialization, word execution and structural diagnostics.
This example supplies the ring and schedule; it does not derive their formation
or introduce a separate phase-update rule. Run after installing the repository.
"""

from __future__ import annotations

from tnfr.sdk import TNFR


def hello_world() -> dict[str, float | int]:
    """Report canonical coherence before and after one declared SDK cycle."""
    network = TNFR.create(6, seed=42).ring()
    initial = network.coherence()
    print("Six-node ring with the SDK's uniform initial triad")
    print(network.results().summary())

    network.evolve(steps=1, sequence="basic_activation")
    final = network.coherence()
    print("After one explicitly requested basic_activation word")
    print(network.results().summary())
    print(network.tetrad().summary())
    print(
        "C and the tetrad are observations of this prepared network. "
        "They do not establish autonomous formation or future stability."
    )
    print("Next: examples/01_foundations/04_operator_sequences.py")
    return {
        "initial_coherence": initial,
        "final_coherence": final,
        "improvement": final - initial,
        "network_size": network.G.number_of_nodes(),
    }


if __name__ == "__main__":
    hello_world()
