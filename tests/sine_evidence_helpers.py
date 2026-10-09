"""Opt-in guards for retained sine evidence; no research imports at collection.

This helper owns execution exclusions only. Each audit retains its independent
mathematics, exact artifact inventory, primitive admission and expected hashes.
"""

import importlib
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

_TARGETS = (
    (
        "tnfr.physics._sine_class_port_prediction",
        (
            "_predict_collective_port_response",
            "_causal_coefficients",
            "_kernel_coefficients",
            "_causal_linear_series",
            "_nonlinear_forcing",
            "_grounded_series",
        ),
    ),
    (
        "tnfr.physics.relational_sine_class_cubic_response",
        (
            "bound_sine_class_cubic_response",
            "_class_cubic_coefficients",
            "_time_coefficients",
            "_coefficient_segment",
        ),
    ),
    (
        "tnfr.mathematics._validated_taylor",
        ("flow_jets", "picard_tube", "validated_box_taylor_step"),
    ),
    ("tnfr.physics._sine_flow", ("_full_sine_field", "_sine_rate_evaluator")),
    ("tnfr.physics._sine_formed_contact", ("_unprobed_handoff",)),
    (
        "tnfr.physics.relational_sine_class_readout",
        (
            "bound_sine_class_four_history_readout",
            "_full_sine_field",
            "validated_box_taylor_step",
        ),
    ),
    (
        "tnfr.physics.relational_sine_class_comparison_readout",
        (
            "bound_sine_class_comparison_readout",
            "bound_sine_class_four_history_readout",
        ),
    ),
    (
        "tnfr.physics.relational_sine_class_port_readout",
        (
            "bound_sine_class_port_readout",
            "_full_sine_field",
            "validated_box_taylor_step",
        ),
    ),
    ("tnfr.research.frozen_source", ("restore_frozen_source",)),
)


@contextmanager
def forbid_sine_regeneration(*, git_root=None, git_base=None):
    """Block generators/restoration, with either no subprocess or pinned Git reads.

    Both Git arguments are required to allow only root/base/blob inspection.
    Imports and patches occur on context entry and are undone even on failure.
    This is an execution guard for explicitly selected audit fixtures, not a
    sandbox or a replacement for the audit's scientific/source checks.
    """
    if (git_root is None) != (git_base is None):
        raise ValueError("read-only Git requires both a root and a full base")
    root = None
    if git_root is not None:
        if not isinstance(git_base, str) or not re.fullmatch(r"[0-9a-f]{40}", git_base):
            raise ValueError("read-only Git requires a full lowercase commit")
        root = Path(git_root).resolve()
    original_popen = subprocess.Popen

    # Resolve every consumer before replacing any owner function. Otherwise a
    # late import can permanently bind a patched owner alias, surviving undo.
    targets = tuple(
        (importlib.import_module(name), attributes) for name, attributes in _TARGETS
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "retained evidence attempted scientific regeneration or restoration"
        )

    def read_only_git(args, *positional, **kwargs):
        command = list(args) if isinstance(args, (tuple, list)) else []
        allowed = command in (
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "cat-file", "-t", git_base],
        )
        allowed |= (
            len(command) == 3
            and command[:2] == ["git", "show"]
            and isinstance(command[2], str)
            and command[2].startswith(git_base + ":src/")
        )
        if (
            not allowed
            or positional
            or set(kwargs) - {"cwd", "stdout", "stderr"}
            or kwargs.get("cwd") is None
            or Path(kwargs["cwd"]).resolve() != root
        ):
            pytest.fail("unexpected subprocess in pinned read-only evidence audit")
        return original_popen(args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for module, attributes in targets:
            for attribute in attributes:
                patch.setattr(module, attribute, forbidden)
        if root is None:
            for name in ("run", "Popen", "check_output"):
                patch.setattr(subprocess, name, forbidden)
        else:
            patch.setattr(subprocess, "Popen", read_only_git)
        yield
