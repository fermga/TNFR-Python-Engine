import math
from benchmarks import tetrad_results_aggregate as agg


def test_validate_and_flatten_schema():
    record = {
        "topology": "ring",
        "n_nodes": 20,
        "seed": 42,
        "precision_mode": "standard",
        "telemetry_density": "low",
        "timings": {
            "phi_s": 0.001,
            "phase_grad": 0.002,
            "phase_curv": 0.003,
            "xi_c": 0.004,
            "tetrad_snapshot": 0.0005,
        },
        "tetrad_values": {
            "phi_s_mean": -2.8,
            "phi_s_std": 0.5,
            "phase_grad_mean": 1.2,
            "phase_grad_std": 0.3,
            "phase_curv_mean": 0.7,
            "phase_curv_std": 0.2,
            "xi_c": None,
        },
        "snapshot_size": 123,
        "total_time": 0.01,
    }
    assert agg.validate_record(record)
    flat = agg.flatten_record(record)
    expected_keys = {
        "topology",
        "n_nodes",
        "seed",
        "precision_mode",
        "telemetry_density",
        "timing_phi_s",
        "timing_phase_grad",
        "timing_phase_curv",
        "timing_xi_c",
        "timing_tetrad_snapshot",
        "phi_s_mean",
        "phi_s_std",
        "phase_grad_mean",
        "phase_grad_std",
        "phase_curv_mean",
        "phase_curv_std",
        "xi_c",
        "snapshot_size",
        "total_time",
    }
    assert expected_keys.issubset(flat.keys())
    assert flat["xi_c"] is None


def test_csv_serialization_precision():
    # Ensure float formatting stable and concise
    value = 0.000123456789
    s = agg._serialize_csv_value(value)
    # Should not expand to scientific with excessive digits
    assert "123456789" not in s  # trimmed
    # numeric parse round-trip tolerance
    assert math.isclose(float(s), value, rel_tol=1e-6)
