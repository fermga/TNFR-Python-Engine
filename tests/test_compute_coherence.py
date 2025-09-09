import math
import pytest

from tnfr.metrics_utils import compute_coherence


def naive_compute_coherence(G):
    dnfr_sum = 0.0
    depi_sum = 0.0
    count = 0
    for _, nd in G.nodes(data=True):
        dnfr_sum += abs(nd.get("dnfr", 0.0))
        depi_sum += abs(nd.get("dEPI", 0.0))
        count += 1
    if count:
        dnfr_mean = dnfr_sum / count
        depi_mean = depi_sum / count
    else:
        dnfr_mean = depi_mean = 0.0
    return 1.0 / (1.0 + dnfr_mean + depi_mean)


def test_compute_coherence_typical(graph_canon):
    G = graph_canon()
    G.add_node(0, dnfr=0.1, dEPI=0.2)
    G.add_node(1, dnfr=0.4, dEPI=0.5)
    result = compute_coherence(G)
    expected = 1.0 / (1.0 + (0.1 + 0.4) / 2 + (0.2 + 0.5) / 2)
    assert result == pytest.approx(expected)


def test_compute_coherence_precision_improved(graph_canon):
    G = graph_canon()
    G.add_node(0, dnfr=1e16)
    for i in range(1, 11):
        G.add_node(i, dnfr=1.0)
    result = compute_coherence(G)
    naive = naive_compute_coherence(G)
    count = G.number_of_nodes()
    expected = 1.0 / (
        1.0
        + math.fsum(abs(nd.get("dnfr", 0.0)) for _, nd in G.nodes(data=True))
        / count
    )
    assert result == expected
    assert abs(result - expected) < abs(naive - expected)


def test_compute_coherence_return_means(graph_canon):
    G = graph_canon()
    G.add_node(0, dnfr=0.1, dEPI=0.2)
    G.add_node(1, dnfr=0.4, dEPI=0.5)
    C, dnfr_mean, depi_mean = compute_coherence(G, return_means=True)
    expected_dnfr = (0.1 + 0.4) / 2
    expected_depi = (0.2 + 0.5) / 2
    expected_C = 1.0 / (1.0 + expected_dnfr + expected_depi)
    assert C == pytest.approx(expected_C)
    assert dnfr_mean == pytest.approx(expected_dnfr)
    assert depi_mean == pytest.approx(expected_depi)
