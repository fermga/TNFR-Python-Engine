import json
import csv
from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork


def test_export_prime_certificates_jsonl(tmp_path):
    net = ArithmeticTNFRNetwork(max_number=50)
    out_file = tmp_path / "prime_certs.jsonl"
    count = net.export_prime_certificates(
        str(out_file),
        delta_nfr_threshold=0.25,
        fmt="jsonl",
        include_components=True,
    )
    assert out_file.exists()
    assert count > 0
    # Read first few lines
    with out_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    assert lines, "No lines written to prime certificates export"
    first = json.loads(lines[0])
    required_keys = {
        "number",
        "delta_nfr",
        "structural_prime",
        "tau",
        "sigma",
        "omega",
    }
    assert required_keys.issubset(first.keys())
    assert isinstance(first["delta_nfr"], float)
    # Components present when include_components=True
    assert "components" in first


def test_export_prime_certificates_csv(tmp_path):
    net = ArithmeticTNFRNetwork(max_number=40)
    out_file = tmp_path / "prime_certs.csv"
    count = net.export_prime_certificates(
        str(out_file),
        delta_nfr_threshold=0.25,
        fmt="csv",
        include_components=False,
    )
    assert out_file.exists()
    assert count > 0
    with out_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) > 1
    header = rows[0]
    assert "components" not in header


def test_export_structural_fields_jsonl(tmp_path):
    net = ArithmeticTNFRNetwork(max_number=60)
    out_file = tmp_path / "fields.jsonl"
    count = net.export_structural_fields(
        str(out_file),
        phase_method="logn",
        fmt="jsonl",
    )
    assert out_file.exists()
    assert count == 59  # numbers 2..60
    with out_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert lines, "No lines written"
    # Last line should be xi_c summary JSON object
    summary = json.loads(lines[-1])
    assert "xi_c_summary" in summary
    # Check a data line
    first_data = json.loads(lines[0])
    for key in ["n", "phi", "phi_grad", "k_phi", "phi_s"]:
        assert key in first_data


def test_export_structural_fields_csv(tmp_path):
    net = ArithmeticTNFRNetwork(max_number=30)
    out_file = tmp_path / "fields.csv"
    count = net.export_structural_fields(
        str(out_file),
        phase_method="logn",
        fmt="csv",
    )
    assert out_file.exists()
    assert count == 29
    with out_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ["n", "phi", "phi_grad", "k_phi", "phi_s"]
    # Find summary header
    assert any(
        r and r[0] == "xi_c_summary_key" for r in rows
    ), "xi_c summary missing"


def test_resonance_seed_reproducibility():
    net = ArithmeticTNFRNetwork(max_number=80)
    # Two histories with same seed must match.
    # Allow tiny numerical noise tolerance.
    h1 = net.resonance_from_primes(steps=3, seed=123, jitter=True)
    h2 = net.resonance_from_primes(steps=3, seed=123, jitter=True)
    assert len(h1) == len(h2)
    for a, b in zip(h1, h2):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            assert abs(a[k] - b[k]) < 1e-12

    # Different seed should produce at least one discrepancy
    # in activation for primes.
    h3 = net.resonance_from_primes(steps=3, seed=999, jitter=True)
    any_diff = False
    for a, c in zip(h1, h3):
        for k in a:
            if abs(a[k] - c[k]) > 1e-12:
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff, (
        "Different seeds produced identical histories; expected divergence"
    )
