
import numpy as np
from typing import Optional

class BBoxEKF:
    """
    Center-state EKF.  Internal state is [cx, cy, vx, vy].
    Corners are only used at input (bbox → center) and output (center → bbox).
    This makes velocity completely immune to bbox shrinkage at occlusion entry.
    """

    DIM_X = 4
    DIM_Z = 2   # we observe (cx, cy) only

    def __init__(self, bbox, process_noise: float = 2.0, meas_noise: float = 5.0,
                 vel_window: int = 100):
        x1, y1, w, h = bbox
        cx = float(x1 + w / 2.0)
        cy = float(y1 + h / 2.0)
        self._bw = float(w)
        self._bh = float(h)

        self.x = np.array([cx, cy, 0., 0.])
        self.P = np.diag([10., 10., 100., 100.]).astype(float)
        self.Q = np.diag([
            process_noise,
            process_noise,
            process_noise * 4,
            process_noise * 4,
        ]).astype(float)
        self.R = np.eye(self.DIM_Z, dtype=float) * meas_noise
        self._H_mat = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ], dtype=float)

    def _compute_jacobian(self, H, H_reliable, cx, cy, new_cx_h, new_cy_h) -> np.ndarray:
        F = np.eye(self.DIM_X, dtype=float)
        F[0, 2] = 1.0
        F[1, 3] = 1.0
        if H is not None and H_reliable:
            denom = H[2, 0] * cx + H[2, 1] * cy + H[2, 2] + 1e-8
            A = (H[0, 0] - new_cx_h * H[2, 0]) / denom
            B = (H[0, 1] - new_cx_h * H[2, 1]) / denom
            C = (H[1, 0] - new_cy_h * H[2, 0]) / denom
            D = (H[1, 1] - new_cy_h * H[2, 1]) / denom
            F[0, 0] = A;   F[0, 1] = B
            F[1, 0] = C;   F[1, 1] = D
        return F

    def predict(self, H: Optional[np.ndarray] = None, H_reliable: bool = False):
        cx, cy = self.x[0], self.x[1]
        if H is not None and H_reliable:
            denom    = H[2, 0] * cx + H[2, 1] * cy + H[2, 2]
            new_cx_h = (H[0, 0] * cx + H[0, 1] * cy + H[0, 2]) / (denom + 1e-8)
            new_cy_h = (H[1, 0] * cx + H[1, 1] * cy + H[1, 2]) / (denom + 1e-8)
        else:
            new_cx_h = cx
            new_cy_h = cy
        self.x[0] = new_cx_h + self.x[2]
        self.x[1] = new_cy_h + self.x[3]
        F      = self._compute_jacobian(H, H_reliable, cx, cy, new_cx_h, new_cy_h)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, bbox):
        x1, y1, w, h = bbox
        cx = float(x1 + w / 2.0)
        cy = float(y1 + h / 2.0)
        self._bw = 0.85 * self._bw + 0.15 * float(w)
        self._bh = 0.85 * self._bh + 0.15 * float(h)
        z     = np.array([cx, cy])
        innov = z - self._H_mat @ self.x
        S     = self._H_mat @ self.P @ self._H_mat.T + self.R
        K     = self.P @ self._H_mat.T @ np.linalg.inv(S)
        self.x     = self.x + K @ innov
        I_KH       = np.eye(self.DIM_X) - K @ self._H_mat
        self.P     = I_KH @ self.P

    def get_bbox(self) -> np.ndarray:
        cx, cy = self.x[0], self.x[1]
        x1 = cx - self._bw / 2.0
        y1 = cy - self._bh / 2.0
        return np.array([int(x1), int(y1),
                         max(1, int(self._bw)),
                         max(1, int(self._bh))], dtype=int)

    def get_velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def get_uncertainty(self) -> float:
        return float(np.sqrt(np.diag(self.P[:2]).mean()))

    def reseed(self, bbox, velocity: np.ndarray):
        x1, y1, w, h = bbox
        self.x[0] = float(x1 + w / 2.0)
        self.x[1] = float(y1 + h / 2.0)
        self.x[2] = float(velocity[0])
        self.x[3] = float(velocity[1])
        self._bw  = float(w)
        self._bh  = float(h)
        self.P    = np.diag([25., 25., 40., 40.]).astype(float)

    def nudge_position(self, bbox):
        x1, y1, w, h = bbox
        self.x[0] = float(x1 + w / 2.0)
        self.x[1] = float(y1 + h / 2.0)
        self._bw  = float(w)
        self._bh  = float(h)
