"""Pure jitter proposals preserve the established deterministic stream."""

from __future__ import annotations

import hashlib
import random
import struct
from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.node import NodeNX
from tnfr.operators.jitter import (
    JitterProgress,
    commit_jitter_proposal,
    propose_jitter_draw,
    random_jitter,
)


_PROGRESS_KEY = "_rng_jitter_progress"
_MASK64 = (1 << 64) - 1


def _legacy_seed_hash(seed: int, key: int) -> int:
    """Independently reproduce the established signed 64-bit seed derivation."""

    payload = struct.pack(">QQ", seed & _MASK64, key & _MASK64)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _legacy_jitter(seed: int, offset: int, draw: int, amplitude: float) -> float:
    node_seed = _legacy_seed_hash(seed, offset)
    return random.Random(_legacy_seed_hash(node_seed, draw)).uniform(
        -amplitude,
        amplitude,
    )


@pytest.mark.parametrize("seed", [0, -1, 2**130 + 9])
@pytest.mark.parametrize("offset", [0, 7, 2**70 + 3])
@pytest.mark.parametrize("draw", [0, 1, 19])
def test_proposal_exactly_matches_the_established_jitter_stream(
    seed: int,
    offset: int,
    draw: int,
) -> None:
    progress = {"seed": seed, "offset": offset, "draws": draw}

    proposal = propose_jitter_draw(
        0.375,
        seed=seed,
        offset=offset,
        progress_state=progress,
    )

    assert proposal.value == _legacy_jitter(seed, offset, draw, 0.375)
    assert proposal.progress_before == JitterProgress(seed, offset, draw)
    assert proposal.progress_after == JitterProgress(seed, offset, draw + 1)
    assert proposal.draw_index == draw
    assert proposal.advances


def test_proposal_is_frozen_and_has_no_input_or_global_rng_side_effects() -> None:
    progress = {"seed": 11, "offset": 4, "draws": 2}
    progress_before = deepcopy(progress)
    global_rng_before = random.getstate()

    proposal = propose_jitter_draw(
        0.2,
        seed=11,
        offset=4,
        progress_state=progress,
    )

    assert progress == progress_before
    assert random.getstate() == global_rng_before
    with pytest.raises(FrozenInstanceError):
        setattr(proposal, "value", 0.0)
    assert proposal.progress_after is not None
    with pytest.raises(FrozenInstanceError):
        setattr(proposal.progress_after, "draws", 99)


@pytest.mark.parametrize(
    ("seed", "offset"),
    [(8, 3), (7, 4)],
    ids=["changed-seed", "changed-offset"],
)
def test_changed_stream_identity_restarts_at_draw_zero(seed: int, offset: int) -> None:
    prior = {"seed": 7, "offset": 3, "draws": 12}

    proposal = propose_jitter_draw(
        0.1,
        seed=seed,
        offset=offset,
        progress_state=prior,
    )

    assert proposal.progress_before == JitterProgress(7, 3, 12)
    assert proposal.progress_after == JitterProgress(seed, offset, 1)
    assert proposal.draw_index == 0
    assert proposal.value == _legacy_jitter(seed, offset, 0, 0.1)
    assert prior == {"seed": 7, "offset": 3, "draws": 12}


def test_zero_amplitude_is_a_nonadvancing_proposal_without_dependencies() -> None:
    invalid_progress = {"unrelated": object()}

    proposal = propose_jitter_draw(
        0.0,
        seed="invalid",
        offset=-1,
        progress_state=invalid_progress,
    )

    assert proposal.value == 0.0
    assert proposal.progress_before is None
    assert proposal.progress_after is None
    assert proposal.draw_index is None
    assert not proposal.advances

    storage = {_PROGRESS_KEY: invalid_progress}
    commit_jitter_proposal(storage, proposal)
    assert storage == {_PROGRESS_KEY: invalid_progress}


@pytest.mark.parametrize(
    "progress",
    [
        "invalid",
        {"seed": 1, "offset": 2},
        {"seed": True, "offset": 2, "draws": 0},
        {"seed": 1, "offset": -1, "draws": 0},
        {"seed": 1, "offset": 2, "draws": True},
    ],
)
def test_nonzero_proposal_rejects_malformed_progress(progress: object) -> None:
    with pytest.raises(ValueError, match="RANDOM_SEED|_rng_jitter_progress"):
        propose_jitter_draw(
            0.1,
            seed=1,
            offset=2,
            progress_state=progress,
        )


def test_random_jitter_previews_and_commits_the_same_proposal() -> None:
    graph = nx.path_graph(3)
    graph.graph["RANDOM_SEED"] = -17
    node = NodeNX(graph, 1)
    offset = node.offset()

    for draw in range(5):
        progress = graph.nodes[1].get(_PROGRESS_KEY)
        proposal = propose_jitter_draw(
            0.25,
            seed=-17,
            offset=offset,
            progress_state=progress,
        )

        assert proposal.value == _legacy_jitter(-17, offset, draw, 0.25)
        assert random_jitter(node, 0.25) == proposal.value
        assert proposal.progress_after is not None
        assert graph.nodes[1][_PROGRESS_KEY] == proposal.progress_after.as_record()


def test_commit_rejects_a_stale_proposal_without_mutation() -> None:
    storage = {_PROGRESS_KEY: {"seed": 5, "offset": 2, "draws": 1}}
    proposal = propose_jitter_draw(
        0.3,
        seed=5,
        offset=2,
        progress_state=storage[_PROGRESS_KEY],
    )
    storage[_PROGRESS_KEY] = {"seed": 5, "offset": 2, "draws": 2}
    before = deepcopy(storage)

    with pytest.raises(RuntimeError, match="stale jitter proposal"):
        commit_jitter_proposal(storage, proposal)

    assert storage == before


class _WriteThenFailOnce(dict[str, object]):
    """Mapping that exposes a post-write failure for rollback verification."""

    fail_next_progress_write: bool = True

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, value)
        if key == _PROGRESS_KEY and self.fail_next_progress_write:
            self.fail_next_progress_write = False
            raise RuntimeError("injected progress commit failure")


@pytest.mark.parametrize("has_prior_progress", [False, True])
def test_failed_commit_restores_the_exact_prior_storage_state(
    has_prior_progress: bool,
) -> None:
    prior = {"seed": 13, "offset": 6, "draws": 4}
    storage = _WriteThenFailOnce()
    if has_prior_progress:
        dict.__setitem__(storage, _PROGRESS_KEY, prior)
        progress_state = prior
    else:
        progress_state = None
    proposal = propose_jitter_draw(
        0.4,
        seed=13,
        offset=6,
        progress_state=progress_state,
    )

    with pytest.raises(RuntimeError, match="injected progress commit failure"):
        commit_jitter_proposal(storage, proposal)

    if has_prior_progress:
        assert storage[_PROGRESS_KEY] is prior
    else:
        assert _PROGRESS_KEY not in storage

    commit_jitter_proposal(storage, proposal)
    assert proposal.progress_after is not None
    assert storage[_PROGRESS_KEY] == proposal.progress_after.as_record()
    assert storage[_PROGRESS_KEY] is not prior
