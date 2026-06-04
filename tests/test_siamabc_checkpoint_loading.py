from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.SiamABC.tracker.tracker_setup import (
    INFERENCE_WEIGHT_PREFIXES,
    load_model,
)


class _ToySiamABC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.neck = nn.Linear(4, 4)
        self.polarized_self_attention = nn.Linear(4, 4)
        self.attention_neck = nn.Linear(4, 4)
        self.connect_model = nn.Linear(4, 2)
        self.iou_head = None
        self.auxiliary_head = nn.Linear(4, 1)


def _required_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().half().clone()
        for key, value in model.state_dict().items()
        if key.startswith(INFERENCE_WEIGHT_PREFIXES)
    }


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("prefixed", [False, True])
def test_load_model_accepts_supported_checkpoint_layouts(
    tmp_path: Path,
    wrapped: bool,
    prefixed: bool,
) -> None:
    source = _ToySiamABC()
    state = _required_state(source)
    if prefixed:
        state = {f"module.{key}": value for key, value in state.items()}
    checkpoint = {"state_dict": state} if wrapped else state
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)

    target = _ToySiamABC().float()
    load_model(
        target,
        str(path),
        strict=False,
        require_inference_weights=True,
    )

    loaded = target.state_dict()
    for key, expected in _required_state(source).items():
        assert loaded[key].dtype == torch.float32
        torch.testing.assert_close(loaded[key], expected.float(), rtol=0, atol=0)


def test_load_model_allows_absent_optional_head(tmp_path: Path) -> None:
    source = _ToySiamABC()
    path = tmp_path / "checkpoint.pth"
    torch.save(_required_state(source), path)

    load_model(
        _ToySiamABC(),
        str(path),
        strict=False,
        require_inference_weights=True,
    )


def test_load_model_rejects_missing_inference_weight(tmp_path: Path) -> None:
    source = _ToySiamABC()
    state = _required_state(source)
    state.pop("neck.weight")
    path = tmp_path / "checkpoint.pth"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match="missing required SiamABC inference weights"):
        load_model(
            _ToySiamABC(),
            str(path),
            strict=False,
            require_inference_weights=True,
        )


def test_load_model_rejects_incompatible_inference_weight(tmp_path: Path) -> None:
    source = _ToySiamABC()
    state = _required_state(source)
    state["connect_model.weight"] = torch.zeros(3, 3)
    path = tmp_path / "checkpoint.pth"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match="missing required SiamABC inference weights"):
        load_model(
            _ToySiamABC(),
            str(path),
            strict=False,
            require_inference_weights=True,
        )
