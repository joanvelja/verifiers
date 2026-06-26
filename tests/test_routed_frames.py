"""Roundtrip tests for the shared routed_experts multipart traversal (issue #76)."""

import numpy as np
import pytest

from verifiers.serve.server._routed_frames import (
    detach_routed_attachments,
    reattach_routed_attachments,
)


def _routed_site(arr: np.ndarray, start: int) -> dict:
    """A routed_experts payload as it appears post-model_dump (raw bytes data)."""
    return {
        "data": arr.tobytes(),
        "shape": list(arr.shape),
        "start": start,
    }


def _step(arr: np.ndarray | None, start: int = 0) -> dict:
    tokens = {
        "prompt_ids": [1, 2, 3],
        "completion_ids": [4, 5],
        "routed_experts": _routed_site(arr, start) if arr is not None else None,
    }
    return {"tokens": tokens}


def _rollout_response(arr: np.ndarray, start: int) -> dict:
    """A single-blob RunRolloutResponse.model_dump() shape."""
    return {
        "success": True,
        "error": None,
        "output": {"trajectory": [_step(arr, start)]},
    }


def _group_response(arrs: list[np.ndarray]) -> dict:
    """A RunGroupResponse.model_dump() shape: outputs is a list of RolloutOutput,
    each a single-step trajectory carrying one routed_experts blob."""
    return {
        "success": True,
        "error": None,
        "outputs": [{"trajectory": [_step(arr, start=0)]} for arr in arrs],
    }


@pytest.fixture
def single_blob_arr() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(7, 4, 3), dtype=np.uint8)


def test_detach_reattach_roundtrip(single_blob_arr: np.ndarray):
    control = _rollout_response(single_blob_arr, start=41)
    original_bytes = control["output"]["trajectory"][0]["tokens"]["routed_experts"][
        "data"
    ]

    lightened, attachments = detach_routed_attachments(control)

    # One blob -> one attachment, dense ordinal 0.
    assert len(attachments) == 1
    desc = lightened["output"]["trajectory"][0]["tokens"]["routed_experts"]
    assert desc["frame"] == 0
    assert "data" not in desc
    assert desc["encoding"] == "raw"
    assert desc["dtype"] == "uint8"
    assert desc["shape"] == [7, 4, 3]
    assert desc["start"] == 41

    rehydrated = reattach_routed_attachments(lightened, attachments)
    site = rehydrated["output"]["trajectory"][0]["tokens"]["routed_experts"]

    # Byte-identical data per site after a full roundtrip.
    assert site["data"] == original_bytes
    assert (
        np.frombuffer(site["data"], dtype=np.dtype(site["dtype"]))
        .reshape(site["shape"])
        .tobytes()
        == single_blob_arr.tobytes()
    )


def test_group_multi_sidecar_distinct_content():
    """G3 (highest value): a G=16 group reply, each rollout's RE array filled with
    its OWN index VALUE (not just shape — shapes collide in real debate). After a
    full detach/reattach roundtrip, each hydrated array must carry its own index,
    proving no cross-wiring between sites."""
    G = 16
    shape = (5, 4, 3)  # identical shape across all rollouts -> shape can't disambiguate
    # rollout i is a uint8 array filled entirely with value i.
    arrs = [np.full(shape, i, dtype=np.uint8) for i in range(G)]
    control = _group_response(arrs)

    lightened, attachments = detach_routed_attachments(control)

    # K = G dense ordinals, one frame per rollout.
    assert len(attachments) == G
    descs = [
        out["trajectory"][0]["tokens"]["routed_experts"] for out in lightened["outputs"]
    ]
    assert [d["frame"] for d in descs] == list(range(G))
    assert all("data" not in d for d in descs)

    rehydrated = reattach_routed_attachments(lightened, attachments)

    for i, out in enumerate(rehydrated["outputs"]):
        site = out["trajectory"][0]["tokens"]["routed_experts"]
        hydrated = np.frombuffer(site["data"], dtype=np.dtype(site["dtype"])).reshape(
            site["shape"]
        )
        # The whole point: rollout i's hydrated array carries value i, not some
        # other rollout's value (the silent-shear nightmare).
        assert np.all(hydrated == i), (
            f"rollout {i} cross-wired: got values {np.unique(hydrated)}"
        )
        np.testing.assert_array_equal(hydrated, arrs[i])


def test_encoding_guard(single_blob_arr: np.ndarray):
    """G5: a descriptor with encoding != 'raw' must fail loud on reattach."""
    control = _rollout_response(single_blob_arr, start=0)
    lightened, attachments = detach_routed_attachments(control)
    lightened["output"]["trajectory"][0]["tokens"]["routed_experts"]["encoding"] = (
        "base64"
    )

    with pytest.raises(ValueError, match="encoding"):
        reattach_routed_attachments(lightened, attachments)


def test_nbytes_mismatch_raises(single_blob_arr: np.ndarray):
    """G1/G5: the only length check is len(frame) == prod(shape)*itemsize. A short
    (truncated/dropped) frame must raise — catches a router that forwarded too few
    bytes. This is NOT a token-count cross-check."""
    control = _rollout_response(single_blob_arr, start=0)
    lightened, attachments = detach_routed_attachments(control)
    # Drop the last byte of the single frame -> length no longer matches shape.
    attachments[0] = attachments[0][:-1]

    with pytest.raises(ValueError, match="length"):
        reattach_routed_attachments(lightened, attachments)


def test_zero_len_step():
    """T-EMPTY: a step with shape[0]==0 emits a 0-byte frame; msgpack + ZMQ allow
    empty frames, and the roundtrip must accept it (len 0 == prod([0,4,3])*1)."""
    empty = np.empty((0, 4, 3), dtype=np.uint8)
    control = _rollout_response(empty, start=0)

    lightened, attachments = detach_routed_attachments(control)
    assert len(attachments) == 1
    assert attachments[0] == b""
    desc = lightened["output"]["trajectory"][0]["tokens"]["routed_experts"]
    assert desc["shape"] == [0, 4, 3]

    rehydrated = reattach_routed_attachments(lightened, attachments)
    site = rehydrated["output"]["trajectory"][0]["tokens"]["routed_experts"]
    assert site["data"] == b""
    hydrated = np.frombuffer(site["data"], dtype=np.dtype(site["dtype"])).reshape(
        site["shape"]
    )
    assert hydrated.shape == (0, 4, 3)
