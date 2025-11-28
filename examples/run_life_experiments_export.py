"""
Run Life Experiments and export results to outputs/life as JSON.

Usage:
    python -m examples.run_life_experiments_export
"""
from __future__ import annotations
import json
from datetime import datetime, UTC
from pathlib import Path


from examples.life_experiments import (
    exp1_life_emergence,
    exp2_self_maintenance_coherence,
    exp3_replication_fidelity,
)


def main() -> None:
    t_threshold, Amax = exp1_life_emergence()
    C_final, C_std = exp2_self_maintenance_coherence()
    Fr = exp3_replication_fidelity()

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "exp1_life_emergence": {
            "threshold_time": t_threshold,
            "A_max": Amax,
            "accept": (t_threshold >= 0.0 and Amax > 1.0),
        },
        "exp2_self_maintenance": {
            "C_final": C_final,
            "C_std_last": C_std,
            "accept": (C_final > 0.6 and C_std < 0.02),
        },
        "exp3_replication_fidelity": {
            "Fr": Fr,
            "accept": (Fr > 0.8),
        },
    }

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "life"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"life_experiments_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
