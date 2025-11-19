import math
from benchmarks import tetrad_scaling_exponents as exp


def make_rows():
    # Synthetic timing: T = k * N^a with a = 0.5 for metric 'timing_phi_s'
    # Other metrics slight variations.
    sizes = [10, 20, 40, 80, 160]
    rows = []
    for seed in [1, 2]:
        for n in sizes:
            base = math.sqrt(n)  # exponent 0.5 pattern
            rows.append(
                {
                    "topology": "ring",
                    "n_nodes": n,
                    "seed": seed,
                    "timing_phi_s": base * 0.01,
                    "timing_phase_grad": base * 0.02,
                    "timing_phase_curv": base * 0.03,
                    "timing_xi_c": base * 0.04,
                    "timing_tetrad_snapshot": base * 0.005,
                }
            )
            rows.append(
                {
                    "topology": "ws",
                    "n_nodes": n,
                    "seed": seed,
                    "timing_phi_s": base * 0.02,
                    "timing_phase_grad": base * 0.025,
                    "timing_phase_curv": base * 0.035,
                    "timing_xi_c": base * 0.05,
                    "timing_tetrad_snapshot": base * 0.007,
                }
            )
    return rows


def test_exponent_fit_close_to_expected():
    rows = make_rows()
    valid = exp.filter_valid(rows)
    summary = exp.build_summary(valid, min_points=4)
    ring_phi = summary["metrics"]["timing_phi_s"]["topologies"]["ring"][
        "exponent"
    ]
    ws_phi = summary["metrics"]["timing_phi_s"]["topologies"]["ws"]["exponent"]
    assert math.isclose(ring_phi, 0.5, rel_tol=0.05)
    assert math.isclose(ws_phi, 0.5, rel_tol=0.05)


def test_determinism_same_input():
    rows = make_rows()
    valid1 = exp.filter_valid(rows)
    valid2 = exp.filter_valid(rows)
    summary1 = exp.build_summary(valid1, min_points=4)
    summary2 = exp.build_summary(valid2, min_points=4)
    assert summary1 == summary2
