"""Phase 1 smoke test — 5 GSM8K problems, end-to-end.

Loads SmolLM2, runs the model on each problem, captures hidden states from
3 layers (early/mid/late), projects to a TNFR graph, computes the canonical
tetrad, and prints a tiny table.

Run:
    python IA/experiments/01_microscope_smoke.py

Expected wall-clock: ~5 minutes on RTX 3060 (fp16, batch=1, max 256 new tokens).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make IA/ importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import re

PROMPT_TEMPLATE = (
    "You are a math tutor. Solve the problem step by step and end with "
    "'The answer is <number>'.\n\nProblem: {q}\n\nSolution:"
)

_ANSWER_RE = re.compile(r"answer is\s*[:\-]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _extract_number(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _gold_number(answer_field: str) -> str | None:
    # GSM8K gold answers end with "#### <number>"
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_field)
    return m.group(1) if m else None


def main(n_problems: int = 5, max_new_tokens: int = 256) -> None:
    import torch
    from datasets import load_dataset

    from IA.probe import load_smollm2, tnfr_telemetry_from_hidden

    print("Loading SmolLM2 ...")
    bundle = load_smollm2()
    print(f"  device={bundle.device} dtype={bundle.dtype} n_layers={bundle.n_layers}")

    print("Loading GSM8K (test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n_problems}]")

    n_layers = bundle.n_layers
    layers = (max(1, n_layers // 6), n_layers // 2, n_layers - 1)
    print(f"Probing layers {layers}\n")

    header = f"{'i':>2} {'ok':>3} {'L':>3} {'E_dens':>7} {'Q_top':>7} {'Phi_s':>9} {'|gphi|':>8} {'K_phi':>8} {'xi_C':>8}"
    print(header)
    print("-" * len(header))

    for i, ex in enumerate(ds):
        q = ex["question"]
        gold = _gold_number(ex["answer"])
        prompt = PROMPT_TEMPLATE.format(q=q)
        inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)

        with torch.no_grad():
            gen = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=bundle.tokenizer.eos_token_id,
            )
            # Re-run a forward pass on the full prompt to grab clean hidden states.
            out = bundle.model(**inputs, output_hidden_states=True)

        decoded = bundle.tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = _extract_number(decoded)
        correct = (pred is not None) and (gold is not None) and (pred == gold)

        telem = tnfr_telemetry_from_hidden(
            out.hidden_states, layers=layers, method="cosine_topk", k=8, seed=0
        )

        for layer_idx, t in telem.items():
            canon = t.get("canonical", {})
            tens = t.get("tensor_invariants", {})

            def _mean(d):
                if isinstance(d, dict) and d:
                    vals = [v for v in d.values() if v == v]  # drop NaN
                    return sum(vals) / len(vals) if vals else float("nan")
                return float(d) if isinstance(d, (int, float)) else float("nan")

            print(
                f"{i:>2} {('Y' if correct else 'N'):>3} {layer_idx:>3} "
                f"{tens.get('energy_density', float('nan')):>7.3f} "
                f"{tens.get('topological_charge', float('nan')):>7.3f} "
                f"{_mean(canon.get('phi_s')):>9.4f} "
                f"{_mean(canon.get('grad_phi')):>8.4f} "
                f"{_mean(canon.get('curv_phi')):>8.4f} "
                f"{canon.get('xi_c', float('nan')):>8.4f}"
            )
        print()

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
