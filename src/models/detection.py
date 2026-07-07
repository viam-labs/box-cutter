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


def inset_endpoints(top_xyz, bottom_xyz, inset_mm=INSET_MM):
    """Pull each 3D endpoint toward the other by inset_mm. Returns (top, bottom).

    If the span is shorter than 2*inset_mm, both collapse to the midpoint.
    """
    t = np.array(top_xyz, dtype=float)
    b = np.array(bottom_xyz, dtype=float)
    span = b - t
    length = float(np.linalg.norm(span))
    if length < 2 * inset_mm:
        mid = tuple((t + b) / 2.0)
        return mid, mid
    u = span / length
    return tuple(t + u * inset_mm), tuple(b - u * inset_mm)


def sample_depth_in_mask(depth_np, mask):
    """Median valid depth (mm) over the box mask, or None if none is available.

    Sampling across the whole box top (not just the centroid pixel) is robust to
    the RealSense dropping depth to 0 on glossy/dark spots (e.g. the tape seam,
    which is exactly where the box centroid tends to land).
    """
    region = depth_np[mask > 0]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def decode_color_and_depth(images: Sequence[NamedImage]):
    """Split a get_images() result into (bgr_color, depth_np_uint16)."""
    color_bgr = None
    depth_np = None
    for img in images:
        if img.mime_type == CameraMimeType.VIAM_RAW_DEPTH:
            depth_np = np.array(img.bytes_to_depth_array(), dtype=np.uint16)
        elif img.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
            buf = np.frombuffer(img.data, dtype=np.uint8)
            color_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if color_bgr is None:
        raise ValueError("camera did not return a color (JPEG/PNG) frame")
    if depth_np is None:
        raise ValueError("camera did not return a depth (VIAM_RAW_DEPTH) frame")
    return color_bgr, depth_np


def detect_box_center(
    bgr,
    hsv_lower=HSV_LOWER,
    hsv_upper=HSV_UPPER,
    min_box_area=MIN_BOX_AREA,
):
    """Find the box center by HSV color segmentation.

    Returns (cX, cY, box_mask) -- the centroid pixel and the filled binary mask of
    the chosen box contour -- or (None, None, None) if no box-colored region passes
    the area threshold.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_box_area:
        return None, None, None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    box_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(box_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return cX, cY, box_mask
