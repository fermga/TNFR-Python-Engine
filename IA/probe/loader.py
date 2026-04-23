"""SmolLM2 loader for Phase 1.

Loads ``HuggingFaceTB/SmolLM2-1.7B-Instruct`` in fp16 on the available GPU,
configured to return all hidden states. Frozen — no training in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    n_layers: int


def load_smollm2(
    device: str | None = None,
    dtype: str = "float16",
    model_id: str = MODEL_ID,
) -> LoadedModel:
    """Load SmolLM2 with hidden-state output enabled.

    Parameters
    ----------
    device : "cuda" | "cpu" | None
        ``None`` auto-detects (cuda if available, else cpu).
    dtype : "float16" | "bfloat16" | "float32"
        Inference dtype. fp16 fits in 12 GB VRAM with margin.
    model_id : str
        HuggingFace repo id.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
    )
