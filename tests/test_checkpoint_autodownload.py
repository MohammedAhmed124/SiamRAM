from __future__ import annotations

from pathlib import Path

import run_inference
from ultralytics.utils import downloads


def _write_valid_checkpoint(path: Path, size: int = 1_100_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


def test_missing_custom_yolo_checkpoint_is_downloaded_to_requested_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weights_path = tmp_path / "checkpoints" / "inference_checkpoint.pth"
    yolo_path = tmp_path / "checkpoints" / "yolo11x.pt"
    downloaded_elsewhere = tmp_path / "downloads" / "yolo11x.pt"

    _write_valid_checkpoint(weights_path)

    def _fake_attempt_download_asset(
        file: str | Path,
        repo: str = "ultralytics/assets",
        release: str = "latest",
        **kwargs,
    ) -> str:
        del file, repo, release, kwargs
        _write_valid_checkpoint(downloaded_elsewhere)
        return str(downloaded_elsewhere)

    monkeypatch.setattr(
        downloads,
        "attempt_download_asset",
        _fake_attempt_download_asset,
    )
    monkeypatch.setattr(
        run_inference,
        "CHECKPOINT_DOWNLOADER",
        tmp_path / "missing_download_checkpoints.py",
    )

    resolved_weights, resolved_yolo = run_inference._ensure_required_checkpoints(
        weights_path=weights_path,
        yolo_path=yolo_path,
    )

    assert resolved_weights == weights_path
    assert resolved_yolo == yolo_path
    assert yolo_path.exists()
    assert yolo_path.stat().st_size >= 1_000_000


def test_missing_custom_yolo_checkpoint_falls_back_to_latest_tag_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weights_path = tmp_path / "checkpoints" / "inference_checkpoint.pth"
    yolo_path = tmp_path / "checkpoints" / "yolo11l.pt"
    _write_valid_checkpoint(weights_path)

    def _failing_attempt_download_asset(*args, **kwargs) -> str:
        del args, kwargs
        raise RuntimeError("simulated default download failure")

    def _fake_get_github_assets(
        repo: str = "ultralytics/assets",
        version: str = "latest",
        retry: bool = False,
    ) -> tuple[str, list[str]]:
        del retry
        assert repo == "ultralytics/assets"
        assert version == "latest"
        return "v9.9.9", ["yolo11l.pt"]

    def _fake_safe_download(
        url: str,
        file: str | Path | None = None,
        **kwargs,
    ):
        del kwargs
        assert "releases/download/v9.9.9/yolo11l.pt" in url
        assert file is not None
        _write_valid_checkpoint(Path(file))
        return Path(file)

    monkeypatch.setattr(
        downloads,
        "attempt_download_asset",
        _failing_attempt_download_asset,
    )
    monkeypatch.setattr(downloads, "get_github_assets", _fake_get_github_assets)
    monkeypatch.setattr(downloads, "safe_download", _fake_safe_download)
    monkeypatch.setattr(
        run_inference,
        "CHECKPOINT_DOWNLOADER",
        tmp_path / "missing_download_checkpoints.py",
    )

    resolved_weights, resolved_yolo = run_inference._ensure_required_checkpoints(
        weights_path=weights_path,
        yolo_path=yolo_path,
    )

    assert resolved_weights == weights_path
    assert resolved_yolo == yolo_path
    assert yolo_path.exists()
    assert yolo_path.stat().st_size >= 1_000_000
