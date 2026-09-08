#!/usr/bin/env python3
"""Example 150 -- Parry measure on the flat grammar automaton.

This example places a *chosen* maximum-entropy Markov policy on the dominant
strongly connected component of the finite flat automaton. The Parry measure is
canonical for that finite labelled graph, but the grammar does not prescribe
that runtime operator-selection policy.

The entropy identity and relative-entropy contraction below are facts about this
Markov chain. They are not a TNFR thermodynamic equilibrium, a grammar-validity
criterion, or an H-theorem for nodal dynamics. Runtime U3 and U6 checks and the
stack-like U5 nesting layer remain outside the chain.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from _flat_grammar_model import (
    ALPHA,
    SHORT,
    build_automaton,
    spectral_radius,
    transfer_matrix,
)


def build_trim():
    """Return trimmed flat states, adjacency multiplicity, and edge labels."""

    states, edges = build_automaton()
    trim, index, matrix = transfer_matrix(states, edges)
    labels = {}
    for state in trim:
        for symbol, next_state in edges.get(state, ()):
            if next_state in index:
                labels.setdefault((index[state], index[next_state]), []).append(symbol)
    return trim, index, matrix, labels


def dominant_scc(matrix):
    """Select an SCC whose internal matrix carries the global spectral radius."""

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(matrix)))
    rows, columns = np.nonzero(matrix)
    graph.add_edges_from(zip(rows.tolist(), columns.tolist()))
    target = spectral_radius(matrix)
    candidates = []
    for component in nx.strongly_connected_components(graph):
        indices = sorted(component)
        submatrix = matrix[np.ix_(indices, indices)]
        candidates.append((spectral_radius(submatrix), indices, submatrix))
    radius, indices, submatrix = max(candidates, key=lambda item: item[0])
    if not np.isclose(radius, target, rtol=1e-10, atol=1e-10):
        raise RuntimeError("no SCC carries the transfer-matrix spectral radius")
    return indices, submatrix, radius


def recurrent_scc(trim, index, matrix):
    """Compatibility wrapper for the previous three-argument helper."""

    del trim, index
    return dominant_scc(matrix)


def parry_measure(matrix):
    """Return aggregated Parry transition matrix and stationary distribution."""

    eigenvalues, right_vectors = np.linalg.eig(matrix)
    right_index = int(np.argmax(eigenvalues.real))
    radius = float(eigenvalues[right_index].real)
    right = np.abs(np.real_if_close(right_vectors[:, right_index]).astype(float))

    left_values, left_vectors = np.linalg.eig(matrix.T)
    left_index = int(np.argmax(left_values.real))
    left = np.abs(np.real_if_close(left_vectors[:, left_index]).astype(float))

    transition = matrix * right[np.newaxis, :]
    transition = transition / (radius * right[:, np.newaxis])
    transition[np.abs(transition) < 1e-15] = 0.0
    stationary = left * right
    stationary = stationary / stationary.sum()

    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("Parry transition rows are not stochastic")
    if not np.allclose(stationary @ transition, stationary, atol=1e-10):
        raise RuntimeError("Parry stationary distribution residual is too large")
    return transition, stationary, right, radius


def _stationary_distribution(transition):
    values, vectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    stationary = np.abs(np.real_if_close(vectors[:, index]).astype(float))
    return stationary / stationary.sum()


def _entropy_components(matrix, transition, stationary):
    state_entropy = 0.0
    choice_entropy = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            probability = transition[i, j]
            if probability <= 0.0:
                continue
            state_entropy -= stationary[i] * probability * np.log(probability)
            multiplicity = int(round(matrix[i, j]))
            if multiplicity > 1:
                choice_entropy += stationary[i] * probability * np.log(multiplicity)
    return state_entropy, choice_entropy


def experiment_1_maximum_entropy_policy(matrix, transition, stationary, radius):
    print("=" * 72)
    print("M1: SELECTED MAXIMUM-ENTROPY POLICY ON THE DOMINANT SCC")
    print("=" * 72)
    state_entropy, choice_entropy = _entropy_components(
        matrix, transition, stationary
    )
    parry_entropy = state_entropy + choice_entropy

    # Comparison policy: choose every outgoing labelled edge uniformly.
    outdegree = matrix.sum(axis=1)
    uniform = matrix / outdegree[:, np.newaxis]
    uniform_stationary = _stationary_distribution(uniform)
    uniform_entropy = float(np.sum(uniform_stationary * np.log(outdegree)))

    print(f"  SCC states                         = {len(matrix)}")
    print(f"  transfer spectral radius           = {radius:.10f}")
    print(f"  Parry labelled-edge entropy rate   = {parry_entropy:.10f} nats/op")
    print(f"  uniform-outgoing-edge entropy rate = {uniform_entropy:.10f} nats/op")
    print(f"  log(radius)                        = {np.log(radius):.10f} nats/op")
    assert np.isclose(parry_entropy, np.log(radius), atol=1e-10)
    assert parry_entropy + 1e-10 >= uniform_entropy
    print("\n  The maximum-entropy statement is conditional on selecting stationary")
    print("  Markov measures supported by this finite labelled component.")
    return state_entropy, choice_entropy


def experiment_2_capacity_split(matrix, transition, stationary, radius):
    print("\n" + "=" * 72)
    print("M2: STATE-TARGET ENTROPY + PARALLEL-LABEL CHOICE ENTROPY")
    print("=" * 72)
    state_entropy, choice_entropy = _entropy_components(
        matrix, transition, stationary
    )
    total = state_entropy + choice_entropy
    print(f"  H(next state | current state) = {state_entropy:.10f} nats/op")
    print(f"  H(label | state transition)   = {choice_entropy:.10f} nats/op")
    print(f"  sum                           = {total:.10f} nats/op")
    print(f"  log(radius)                   = {np.log(radius):.10f} nats/op")
    print(f"  numerical residual            = {abs(total - np.log(radius)):.2e}")
    assert np.isclose(total, np.log(radius), atol=1e-10)
    print("\n  This is the entropy chain rule for parallel labelled edges. It is")
    print("  information in the selected symbolic process, not TNFR energy.")


def experiment_3_relative_entropy(transition, stationary):
    print("\n" + "=" * 72)
    print("M3: RELATIVE-ENTROPY CONTRACTION FOR THE SELECTED MARKOV KERNEL")
    print("=" * 72)
    distribution = np.zeros(len(stationary))
    distribution[int(np.argmax(stationary))] = 1.0
    divergences = []
    for _ in range(41):
        mask = distribution > 0.0
        divergence = float(
            np.sum(distribution[mask] * np.log(distribution[mask] / stationary[mask]))
        )
        divergences.append(divergence)
        distribution = distribution @ transition

    for step in (0, 1, 2, 5, 10, 20, 40):
        print(f"  t={step:>2}: D(p_t || pi) = {divergences[step]:.10f}")
    monotone = all(
        right <= left + 1e-11
        for left, right in zip(divergences, divergences[1:])
    )
    print(f"  non-increasing over sampled steps: {monotone}")
    assert monotone
    print("\n  This is Markov data processing for the finite selected kernel")
    print("  relative to its stationary Parry measure. It does not show that an")
    print("  engine trajectory follows this kernel or prove nodal C(t) convergence.")


def experiment_4_operator_frequencies(component, transition, stationary, labels):
    print("\n" + "=" * 72)
    print("M4: LABEL FREQUENCIES UNDER THE SELECTED POLICY")
    print("=" * 72)
    local = {
        global_index: local_index
        for local_index, global_index in enumerate(component)
    }
    frequencies = {symbol: 0.0 for symbol in ALPHA}
    for global_i in component:
        local_i = local[global_i]
        for global_j in component:
            local_j = local[global_j]
            edge_labels = labels.get((global_i, global_j), ())
            if not edge_labels:
                continue
            share = transition[local_i, local_j] / len(edge_labels)
            for symbol in edge_labels:
                frequencies[symbol] += stationary[local_i] * share
    total = sum(frequencies.values())
    for symbol in frequencies:
        frequencies[symbol] /= total
    for symbol in sorted(ALPHA, key=lambda item: -frequencies[item]):
        print(f"  {SHORT[symbol]:7s} {100.0 * frequencies[symbol]:7.3f}%")
    print("\n  Frequencies depend on the Parry policy and flat automaton; they are")
    print("  not universal operator frequencies or empirical TNFR telemetry.")


def main():
    print("\n" + "#" * 72)
    print("# Example 150 - Parry Measure on the Flat Grammar Automaton")
    print("#" * 72 + "\n")
    _trim, _index, full_matrix, labels = build_trim()
    component, matrix, radius = dominant_scc(full_matrix)
    transition, stationary, _right, parry_radius = parry_measure(matrix)
    if not np.isclose(radius, parry_radius, atol=1e-10):
        raise RuntimeError("inconsistent Perron roots")
    experiment_1_maximum_entropy_policy(matrix, transition, stationary, radius)
    experiment_2_capacity_split(matrix, transition, stationary, radius)
    experiment_3_relative_entropy(transition, stationary)
    experiment_4_operator_frequencies(component, transition, stationary, labels)
    print("\n  Scope: a selected stochastic process on the finite flat model.")
    print("  U3, nested/hierarchy-aware U5, U6, and nodal physics require separate")
    print("  stateful evidence.")


if __name__ == "__main__":
    main()
