"""
NVDEC video reader: a drop-in replacement for cv2.VideoCapture that decodes
H.264/MP4 on the GPU's fixed-function NVDEC block via NVIDIA PyNvVideoCodec.

Why this exists
---------------
On a 4K clip (cows is 3840x2160, sheeps is 2160x4096) software H.264 decode
through cv2.VideoCapture tops out around ~50 fps and becomes the bottleneck of
the whole tracking run — the GPU tracker can process frames far faster than the
CPU can hand them over. Routing the decode through NVDEC removes that wall and
gives a ~4-5x decode speed-up on those clips, so the wall-clock FPS approaches
the tracker-only rate instead of being pinned to the decoder.

Design goals
------------
This class exposes ONLY the subset of the cv2.VideoCapture API that the tracking
pipeline (predictor.run_tracker) actually uses — isOpened(), read(), get() for
the frame count, and release() — and makes each call behave like its cv2
counterpart from the caller's point of view: read() returns one BGR uint8
HxWx3 C-contiguous frame per call, in display order, exactly one row per source
frame, with the trailing/flush frames drained so the total count matches a
software decode of the same file. That lets _open_video hand back either object
and the rest of run_tracker stays untouched.

Safety / fallback
-----------------
Construction never raises. If PyNvVideoCodec cannot be imported, the GPU cannot
be initialised, the demuxer/decoder cannot be created, or the stream's pixel
layout is not the tightly-packed NV12 this reader expects, the object simply
reports isOpened() == False and the caller falls back to plain cv2.VideoCapture.
The pipeline therefore degrades to CPU decode rather than crashing whenever
hardware decode is unavailable.

Decode path
-----------
We use the LOW-LEVEL PyNvVideoCodec API (CreateDemuxer + CreateDecoder +
Decode()), NOT the high-level SimpleDecoder/ThreadedDecoder. The high-level
classes hard-crash the process (native exit 127, no Python exception) on the
4096-wide DCI-4K clip, whereas the low-level path decodes every tested clip
correctly. CreateDecoder runs with the default latency, which emits frames in
DISPLAY order (B-frame reordering is handled internally). Iterating the demuxer
to StopIteration already drains every trailing frame, so NO explicit decoder
flush is required (and PyNvDecoder 2.1.0 exposes none) — the frame count matches
a cv2 software decode exactly.

Colour matching (why NV12 + OpenCV, not NVDEC's RGB)
----------------------------------------------------
We ask NVDEC for its NATIVE output (NV12) and do the YUV->BGR conversion on the
host with cv2.cvtColor(..., COLOR_YUV2BGR_NV12). This is deliberate: NVDEC's own
on-GPU RGB conversion (OutputColorType.RGB) defaults to the BT.709 matrix, while
cv2.VideoCapture's FFmpeg/swscale path hardcodes BT.601 limited-range and
ignores the stream's colour VUI. Letting NVDEC convert therefore produced a
~2.3 LSB mean difference vs the CPU decoder, which perturbed the tracker enough
to measurably move the benchmark score. cv2.COLOR_YUV2BGR_NV12 uses the SAME
BT.601 matrix as cv2.VideoCapture, halving the difference to ~1.1 LSB. The cost
is a host-side colour convert (and pulling NV12, which is 1.5 bytes/px vs 3 for
RGB, so half the PCIe traffic) — decode-to-BGR still runs at ~200-250 fps on 4K,
well above the tracker's throughput, so it stays hidden behind compute.

Colour-accuracy caveat
----------------------
NVDEC's decode and FFmpeg's (cv2's) do not produce bit-identical BGR even with
the matrix matched: the residual ~1 LSB comes from chroma-upsampling and
rounding differences (swscale vs OpenCV) that no PyNvVideoCodec setting can
reproduce. A tracking run with NVDEC is therefore NOT numerically identical to
one with cv2. That is why the master switch (runtime.use_nvdec) stays opt-in and
default-false: enable it for throughput, leave it off when bit-exact parity with
the CPU-decode baseline matters.

Portability
-----------
On Windows the bundled native module links against cudart64_12.dll, which is not
on the default DLL search path when CUDA_PATH points at a CUDA 13 toolkit, so the
import fails out of the box. _prepare_windows_dll_search() adds the CUDA 12.x bin
directory and torch's lib directory (which also ships cudart64_12.dll) to the
search path before the import. On Linux this is a no-op — the manylinux wheel's
RPATH and the system loader already find the bundled FFmpeg/CUDA runtime — and
the whole block is wrapped in try/except so a missing toolkit never raises.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:  # cv2 is already a hard dependency of the pipeline; we only need its enums.
    import cv2
except Exception:  # pragma: no cover - cv2 is always present in this project.
    cv2 = None  # type: ignore[assignment]

_CAP_PROP_FRAME_COUNT = int(getattr(cv2, "CAP_PROP_FRAME_COUNT", 7)) if cv2 is not None else 7
_CAP_PROP_FRAME_WIDTH = int(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)) if cv2 is not None else 3
_CAP_PROP_FRAME_HEIGHT = int(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)) if cv2 is not None else 4


def _prepare_windows_dll_search() -> None:
    """
    Make the PyNvVideoCodec native module loadable on Windows.

    The bundled .pyd links against cudart64_12.dll (CUDA 12 runtime) plus the
    wheel's own FFmpeg DLLs. When CUDA_PATH points at a CUDA 13 toolkit (whose
    runtime is cudart64_13.dll) that DLL is not on the search path and the import
    raises "DLL load failed". We pre-register two directories that are known to
    contain cudart64_12.dll on this kind of box:

      * a CUDA 12.x toolkit bin directory (several common install locations), and
      * torch's own lib directory, which ships cudart64_12.dll as a belt-and-
        braces fallback when no CUDA 12 toolkit is installed.

    Every step is best-effort: a missing directory is skipped and any error is
    swallowed, because failing to add a path here simply lets the subsequent
    import fail, which the caller already treats as "NVDEC unavailable". On any
    non-Windows platform this function returns immediately and changes nothing.
    """
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):  # Python <3.8 on Windows; defensive.
        return

    candidate_dirs: list[str] = []

    for env_key in ("CUDA_PATH_V12_8", "CUDA_PATH_V12_6", "CUDA_PATH_V12_4", "CUDA_PATH"):
        root = os.environ.get(env_key)
        if root:
            candidate_dirs.append(os.path.join(root, "bin"))

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
    for minor in ("v12.8", "v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0"):
        candidate_dirs.append(os.path.join(cuda_root, minor, "bin"))

    try:
        import torch  # noqa: PLC0415 - imported lazily, only on Windows.

        candidate_dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass

    seen: set[str] = set()
    for directory in candidate_dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        try:
            if os.path.isdir(directory):
                os.add_dll_directory(directory)
        except Exception:
            continue


class NvdecVideoCapture:
    """
    Minimal cv2.VideoCapture-compatible reader backed by NVDEC.

    Only the four methods run_tracker relies on are implemented: isOpened(),
    read(), get() and release(). read() yields one BGR uint8 HxWx3 frame per
    call in display order and drains the decoder at end-of-stream so the total
    number of frames returned equals a cv2 software decode of the same file.
    """

    def __init__(self, path: str, gpu_id: int = 0) -> None:
        """
        Open ``path`` for hardware decoding on GPU ``gpu_id``.

        Never raises: any failure (missing library, no CUDA device, unreadable
        file, unexpected pixel layout) leaves the object in a closed state so
        isOpened() returns False and the caller falls back to cv2.VideoCapture.
        """
        self._path = str(path)
        self._gpu_id = int(gpu_id)
        self._opened = False

        self._nvc = None            # the PyNvVideoCodec module
        self._torch = None          # torch, used for the zero-copy surface read
        self._demuxer = None        # packet iterator over the container
        self._decoder = None        # NVDEC decoder
        self._packets = None        # iterator yielding compressed packets
        self._buffer: list[np.ndarray] = []   # decoded BGR frames awaiting read()
        self._eos = False           # demuxer exhausted (no separate flush needed)
        self._frame_count = 0       # CAP_PROP_FRAME_COUNT hint (0 = unknown)
        self._width = 0             # frame width, needed to reshape the NV12 buffer
        self._height = 0            # frame height
        self._nv12_rows = 0         # = height * 3 // 2 (Y plane + interleaved UV)

        try:
            self._open()
            self._opened = True
        except Exception:
            self._opened = False
            self._release_handles()


    def _open(self) -> None:
        """Import the library, create the demuxer/decoder, and prime the first frame."""
        _prepare_windows_dll_search()

        import PyNvVideoCodec as nvc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._nvc = nvc
        self._torch = torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; cannot use NVDEC.")

        self._demuxer = nvc.CreateDemuxer(self._path)
        self._decoder = nvc.CreateDecoder(
            gpuid=self._gpu_id,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.NATIVE,
        )
        self._packets = iter(self._demuxer)
        count, width, height = self._probe_stream_info()
        if width <= 0 or height <= 0:
            raise RuntimeError("could not determine video dimensions for NV12 reshape")
        self._frame_count = count
        self._width = int(width)
        self._height = int(height)
        self._nv12_rows = self._height * 3 // 2
        self._fill_buffer()
        if not self._buffer and self._eos:
            raise RuntimeError("decoder produced no frames")

    def _probe_stream_info(self) -> tuple[int, int, int]:
        """
        Return ``(frame_count, width, height)`` from the container header.

        Uses a lightweight cv2 probe (which reads the MP4 header, not the pixels)
        and never raises. A 0 in any field means "unknown"; an unknown width or
        height makes _open fall back to cv2 (we cannot reshape NV12 without dims).
        """
        if cv2 is None:
            return 0, 0, 0
        try:
            cap = cv2.VideoCapture(self._path)
            count = int(cap.get(_CAP_PROP_FRAME_COUNT))
            width = int(cap.get(_CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(_CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return max(0, count), max(0, width), max(0, height)
        except Exception:
            return 0, 0, 0


    def _surface_to_bgr(self, frame) -> np.ndarray:
        """
        Convert one decoded NVDEC NV12 surface to a host BGR uint8 HxWx3 array.

        ``torch.from_dlpack(frame)`` is a zero-copy view of the NV12 surface in
        GPU memory (the auditor measured this ~18% faster than, and bit-identical
        to, torch.as_tensor(frame.cuda()[0], ...)). We copy it to host as a
        tightly-packed (H*3/2, W) buffer and let OpenCV apply the BT.601
        YUV->BGR conversion that matches cv2.VideoCapture. The .cpu() copy
        produces a fresh host array, so the result is safe to keep even though
        NVDEC reuses its surface pool for the next frame.
        """
        nv12 = self._torch.from_dlpack(frame).reshape(-1).cpu().numpy()
        expected = self._height * self._width * 3 // 2
        if nv12.size != expected:
            raise ValueError(
                f"NV12 buffer size {nv12.size} != expected {expected} "
                f"for {self._width}x{self._height}"
            )
        nv12 = nv12.reshape(self._nv12_rows, self._width)
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

    def _fill_buffer(self) -> None:
        """
        Decode until at least one frame is buffered, or end-of-stream.

        One compressed packet can yield 0..N display frames, so we keep feeding
        packets until Decode() hands back at least one frame. When the packet
        iterator is exhausted we mark end-of-stream: iterating the demuxer to
        StopIteration already drains the decoder's reorder buffer (trailing
        B-frames), so the total frame count matches a cv2 software decode exactly
        and no explicit flush call is needed (PyNvDecoder 2.1.0 exposes none).
        """
        if self._eos:
            return

        while not self._buffer:
            try:
                packet = next(self._packets)
            except StopIteration:
                self._eos = True
                break
            for frame in self._decoder.Decode(packet):
                self._buffer.append(self._surface_to_bgr(frame))


    def isOpened(self) -> bool:
        """Return True when the decoder is ready to produce frames."""
        return bool(self._opened)

    def read(self):
        """
        Return ``(True, frame)`` for the next BGR uint8 HxWx3 frame in display
        order, or ``(False, None)`` once the stream is exhausted.

        Mirrors cv2.VideoCapture.read(): one frame per call, frames buffered
        internally because a single packet may decode to several (or zero)
        frames. Never raises — a decode error is reported as end-of-stream so the
        background reader thread in run_tracker terminates cleanly.
        """
        if not self._opened:
            return False, None

        if not self._buffer:
            try:
                self._fill_buffer()
            except Exception:
                self._eos = True
                return False, None

        if not self._buffer:
            return False, None

        return True, self._buffer.pop(0)

    def get(self, prop_id: int) -> float:
        """
        Return the value of a cv2 capture property.

        Only CAP_PROP_FRAME_COUNT / WIDTH / HEIGHT are meaningfully supported
        (the count backs the progress-bar hint in run_tracker); every other
        property returns 0.0, the same "unknown" sentinel cv2 uses for
        properties it cannot report.
        """
        prop = int(prop_id)
        if prop == _CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == _CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == _CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def release(self) -> None:
        """Free the decoder, demuxer and any buffered frames. Idempotent."""
        self._opened = False
        self._release_handles()

    def _release_handles(self) -> None:
        """Drop all native handles and buffers. Safe to call more than once."""
        self._buffer = []
        self._packets = None
        self._decoder = None
        self._demuxer = None
        self._eos = True
