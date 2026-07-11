from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import predictor


def test_gstreamer_path_is_quoted() -> None:
    quoted = predictor._quote_gstreamer_path("folder with spaces/video.mp4")
    assert quoted.startswith('"')
    assert quoted.endswith('"')
    assert "folder with spaces" in quoted


def test_jetson_capture_validates_and_preserves_first_frame() -> None:
    first = np.zeros((8, 12, 3), dtype=np.uint8)
    second = np.ones((8, 12, 3), dtype=np.uint8)
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, first), (True, second)]
    capture.get.return_value = 2.0

    with patch.object(predictor.cv2, "VideoCapture", return_value=capture) as open_cap:
        wrapped = predictor._open_jetson_gstreamer("video with spaces.mp4")

    assert wrapped is not None
    pipeline = open_cap.call_args.args[0]
    assert "nvv4l2decoder num-extra-surfaces=0" in pipeline
    assert "queue max-size-buffers=2" in pipeline
    assert "appsink sync=false drop=false max-buffers=1" in pipeline
    assert 'filesrc location="' in pipeline
    ok, frame = wrapped.read()
    assert ok and frame is first
    ok, frame = wrapped.read()
    assert ok and frame is second
    assert wrapped.get(7) == 2.0
    wrapped.release()
    capture.release.assert_called_once()


def test_jetson_capture_rejects_pipeline_that_cannot_decode() -> None:
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)

    with patch.object(predictor.cv2, "VideoCapture", return_value=capture):
        assert predictor._open_jetson_gstreamer("broken.mp4") is None

    capture.release.assert_called_once()
