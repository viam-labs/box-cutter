"""Pure box-detection and cut-geometry helpers ported from box-point-test/main.py.

These functions do no I/O (no camera calls, no file writes, no motion). The
Control service passes configured tuning values in; the module-level constants
below are only defaults, matching the original script.
"""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from viam.media.video import CameraMimeType, NamedImage

# --- Box detection tuning defaults (see design doc) ---------------------------
HSV_LOWER = (10, 60, 80)
HSV_UPPER = (35, 255, 255)
MIN_BOX_AREA = 5000
INSET_MM = 8
MIN_SEAM_LEN_PX = 60
SEAM_DARK_V_MAX = 80


def deproject(u, v, z, intr):
    """Pinhole deprojection of pixel (u, v) at depth z (mm) -> (x, y, z) mm."""
    x = (u - intr.center_x_px) * z / intr.focal_x_px
    y = (v - intr.center_y_px) * z / intr.focal_y_px
    return x, y, z
