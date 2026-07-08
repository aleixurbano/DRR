from reflect.core.constants import (
    REAL_FOCAL_LENGTH_X,
    REAL_FOCAL_LENGTH_Y,
    REAL_PRINCIPAL_POINT_X,
    REAL_PRINCIPAL_POINT_Y,
    REAL_SKEW,
    D435I_WIDTH,
    D435I_HEIGHT,
)
from dataclasses import dataclass

@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for a single calibrated RGB stream."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    skew: float = 0.0

def real_k() -> CameraIntrinsics:
    """Return the calibrated D435i intrinsics from `reflect.core.constants`."""
    return CameraIntrinsics(
        fx=REAL_FOCAL_LENGTH_X,
        fy=REAL_FOCAL_LENGTH_Y,
        cx=REAL_PRINCIPAL_POINT_X,
        cy=REAL_PRINCIPAL_POINT_Y,
        skew=REAL_SKEW,
        width=D435I_WIDTH,
        height=D435I_HEIGHT,
    )