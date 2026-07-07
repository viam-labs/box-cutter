# Box Cutter `control` Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port box-center detection and cut-motion planning from the standalone script `../box-point-test/main.py` into the `viam-labs:box-cutter:control` Generic service, exposed through three `do_command` actions (`find_center`, `move_to_center`, `full_cut`).

**Architecture:** Pure vision/geometry functions live in `src/models/detection.py` (no I/O, unit-tested). The `Control` resource in `src/models/control.py` reads config into a settings dataclass, resolves `camera`/`arm`/`motion` dependencies, and dispatches `do_command` to typed async methods. Frame transforms use `motion.get_pose(..., supplemental_transforms=[...])` instead of `RobotClient.transform_pose` (which is unavailable inside a module).

**Tech Stack:** Python 3.10+, `viam-sdk`, OpenCV (`opencv-python-headless`), NumPy, pytest.

**Design doc:** `docs/plans/2026-07-07-box-cutter-control-design.md`

---

## Conventions

- All commands run from the repo root `/Users/nick.hehr/src/box-cutter`.
- Python interpreter is the module venv: `venv/bin/python` (created in Task 0).
- Tests: `venv/bin/python -m pytest <path> -v`.
- Commit after every green step. Use present-tense conventional-commit messages.
- Follow @superpowers:test-driven-development for every task with a test: write the failing test, watch it fail, implement minimally, watch it pass, commit.

---

## Task 0: Dependencies and dev environment

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`

**Step 1: Update `requirements.txt`**

Replace the contents with:

```
viam-sdk==0.79.1

typing-extensions
numpy
opencv-python-headless
```

(`opencv-python-headless` — the module runs on a robot with no display; it still provides `cv2`.)

**Step 2: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest
```

**Step 3: Create `pytest.ini`**

```ini
[pytest]
pythonpath = src
testpaths = tests
```

(`pythonpath = src` lets tests do `from models.detection import ...`, matching how `src/main.py` imports `from models.control import ...`.)

**Step 4: Create `tests/__init__.py`** (empty file).

**Step 5: Create the venv and install dev deps**

Run:
```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements-dev.txt
```
Expected: installs succeed; `venv/bin/python -c "import cv2, numpy, viam; print('ok')"` prints `ok`.

**Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini tests/__init__.py
git commit -m "chore: add opencv/numpy deps and pytest dev setup"
```

Note: `venv/` and `.installed` are already covered by `.gitignore`. Do not commit them.

---

## Task 1: `deproject` (pure pinhole deprojection)

**Files:**
- Create: `src/models/detection.py`
- Test: `tests/test_detection.py`

**Step 1: Write the failing test**

```python
# tests/test_detection.py
import numpy as np
import pytest

from models.detection import deproject


class _Intr:
    """Minimal stand-in for viam intrinsic_parameters."""
    def __init__(self, fx, fy, cx, cy):
        self.focal_x_px = fx
        self.focal_y_px = fy
        self.center_x_px = cx
        self.center_y_px = cy


def test_deproject_center_pixel_is_on_axis():
    intr = _Intr(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    x, y, z = deproject(320, 240, 1000.0, intr)
    assert (x, y, z) == (0.0, 0.0, 1000.0)


def test_deproject_offset_pixel():
    intr = _Intr(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    x, y, z = deproject(320 + 60, 240, 1000.0, intr)
    # (u - cx) * z / fx = 60 * 1000 / 600 = 100
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(0.0)
    assert z == 1000.0
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.detection'`.

**Step 3: Create `src/models/detection.py` with the module header and `deproject`**

```python
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
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS (2 passed).

**Step 5: Commit**

```bash
git add src/models/detection.py tests/test_detection.py
git commit -m "feat: add deproject helper to detection module"
```

---

## Task 2: `inset_endpoints`

**Files:**
- Modify: `src/models/detection.py`
- Test: `tests/test_detection.py`

**Step 1: Add failing tests**

```python
from models.detection import inset_endpoints


def test_inset_endpoints_pulls_inward():
    top, bottom = inset_endpoints((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), inset_mm=10)
    assert top == pytest.approx((10.0, 0.0, 0.0))
    assert bottom == pytest.approx((90.0, 0.0, 0.0))


def test_inset_endpoints_collapses_when_too_short():
    # span (10mm) < 2 * inset (2*8=16) -> both collapse to midpoint
    top, bottom = inset_endpoints((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), inset_mm=8)
    assert top == pytest.approx((5.0, 0.0, 0.0))
    assert bottom == pytest.approx((5.0, 0.0, 0.0))
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — `ImportError: cannot import name 'inset_endpoints'`.

**Step 3: Add implementation to `src/models/detection.py`**

```python
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
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS (4 passed).

**Step 5: Commit**

```bash
git add src/models/detection.py tests/test_detection.py
git commit -m "feat: add inset_endpoints helper"
```

---

## Task 3: `sample_depth_in_mask`

**Files:**
- Modify: `src/models/detection.py`
- Test: `tests/test_detection.py`

**Step 1: Add failing tests**

```python
from models.detection import sample_depth_in_mask


def test_sample_depth_median_ignores_zeros():
    depth = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
    mask = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    # valid values: 1000, 2000, 3000 -> median 2000
    assert sample_depth_in_mask(depth, mask) == pytest.approx(2000.0)


def test_sample_depth_returns_none_when_all_invalid():
    depth = np.zeros((2, 2), dtype=np.uint16)
    mask = np.full((2, 2), 255, dtype=np.uint8)
    assert sample_depth_in_mask(depth, mask) is None
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — cannot import `sample_depth_in_mask`.

**Step 3: Add implementation**

```python
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
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS (6 passed).

**Step 5: Commit**

```bash
git add src/models/detection.py tests/test_detection.py
git commit -m "feat: add sample_depth_in_mask helper"
```

---

## Task 4: `decode_color_and_depth`

**Files:**
- Modify: `src/models/detection.py`
- Test: `tests/test_detection.py`

**Step 1: Add failing tests**

```python
from models.detection import decode_color_and_depth


class _FakeNamedImage:
    """Enough of NamedImage for decode_color_and_depth."""
    def __init__(self, mime_type, data=b"", depth=None):
        self.mime_type = mime_type
        self.data = data
        self._depth = depth

    def bytes_to_depth_array(self):
        return self._depth


def _jpeg_bytes(bgr):
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def test_decode_color_and_depth_splits_streams():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:] = (0, 0, 200)  # red-ish so decode is non-trivial
    depth = np.full((4, 4), 500, dtype=np.uint16)
    images = [
        _FakeNamedImage(CameraMimeType.JPEG, data=_jpeg_bytes(bgr)),
        _FakeNamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ]
    color_out, depth_out = decode_color_and_depth(images)
    assert color_out.shape == (4, 4, 3)
    assert depth_out.shape == (4, 4)
    assert depth_out.dtype == np.uint16


def test_decode_raises_without_color():
    depth = np.full((4, 4), 500, dtype=np.uint16)
    images = [_FakeNamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth)]
    with pytest.raises(ValueError, match="color"):
        decode_color_and_depth(images)


def test_decode_raises_without_depth():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    images = [_FakeNamedImage(CameraMimeType.JPEG, data=_jpeg_bytes(bgr))]
    with pytest.raises(ValueError, match="depth"):
        decode_color_and_depth(images)
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — cannot import `decode_color_and_depth`.

**Step 3: Add implementation** (ported verbatim from the script)

```python
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
        raise ValueError("camera did not return a VIAM_RAW_DEPTH frame")
    return color_bgr, depth_np
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS (9 passed).

**Step 5: Commit**

```bash
git add src/models/detection.py tests/test_detection.py
git commit -m "feat: add decode_color_and_depth helper"
```

---

## Task 5: `detect_box_center` (debug image write removed)

**Files:**
- Modify: `src/models/detection.py`
- Test: `tests/test_detection.py`

**Step 1: Add failing tests**

```python
from models.detection import detect_box_center


def _tan_box_image(w=640, h=480, box=(220, 160, 200, 160)):
    """Neutral-gray background with one solid tan (HSV ~20,150,200) rectangle."""
    bgr = np.full((h, w, 3), 128, dtype=np.uint8)  # neutral gray
    x, y, bw, bh = box
    # A tan/orange BGR that lands inside HSV_LOWER..HSV_UPPER.
    bgr[y:y + bh, x:x + bw] = (60, 140, 200)  # BGR
    return bgr, (x + bw // 2, y + bh // 2)


def test_detect_box_center_finds_rectangle_center():
    bgr, (cx_true, cy_true) = _tan_box_image()
    cx, cy, mask = detect_box_center(bgr)
    assert cx is not None and cy is not None
    assert abs(cx - cx_true) <= 5
    assert abs(cy - cy_true) <= 5
    assert mask.shape == bgr.shape[:2]
    assert mask.max() == 255


def test_detect_box_center_rejects_small_speck():
    bgr, _ = _tan_box_image(box=(10, 10, 20, 20))  # 400 px^2 << MIN_BOX_AREA
    cx, cy, mask = detect_box_center(bgr)
    assert cx is None and cy is None and mask is None


def test_detect_box_center_none_when_no_box_color():
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)  # all gray, no tan
    cx, cy, mask = detect_box_center(bgr)
    assert cx is None and cy is None and mask is None
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — cannot import `detect_box_center`.

**Step 3: Add implementation** (ported; the `cv2.imwrite`/debug-draw block from the script is removed, and tuning values become parameters)

```python
def detect_box_center(
    bgr,
    hsv_lower=HSV_LOWER,
    hsv_upper=HSV_UPPER,
    min_box_area=MIN_BOX_AREA,
):
    """Find the box center by HSV color segmentation.

    Returns (cX, cY, box_mask) — the centroid pixel and the filled binary mask of
    the chosen box contour — or (None, None, None) if no box-colored region passes
    the area threshold.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)   # kill specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)  # bridge seam

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_box_area:
        return None, None, None

    # Centroid from image moments = true center of mass (correct even when rotated).
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Restrict the returned mask to just the chosen box contour so depth sampling
    # stays on the box.
    box_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(box_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return cX, cY, box_mask
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS (12 passed).

**Step 5: Commit**

```bash
git add src/models/detection.py tests/test_detection.py
git commit -m "feat: add detect_box_center (no debug image I/O)"
```

---

## Task 6: `_pick_seam_axis` + `find_seam_edges`

**Files:**
- Modify: `src/models/detection.py`
- Test: `tests/test_detection.py`

**Step 1: Add a failing test**

```python
from models.detection import find_seam_edges


def test_find_seam_edges_returns_endpoints_along_dark_seam():
    # Tan box with a vertical dark seam down the middle.
    h, w = 480, 640
    bgr = np.full((h, w, 3), 128, dtype=np.uint8)
    bgr[140:340, 240:400] = (60, 140, 200)     # tan box
    bgr[140:340, 316:324] = (10, 10, 10)        # dark vertical seam
    _, _, mask = detect_box_center(bgr)
    assert mask is not None
    center = (320, 240)
    result = find_seam_edges(mask, center, bgr)
    assert result is not None
    top_px, bottom_px, angle_deg = result
    # Seam is vertical -> endpoints share x ~= center x, differ in y.
    assert abs(top_px[0] - center[0]) <= 10
    assert abs(bottom_px[0] - center[0]) <= 10
    assert abs(top_px[1] - bottom_px[1]) > 50


def test_find_seam_edges_none_on_empty_mask():
    mask = np.zeros((480, 640), dtype=np.uint8)
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    assert find_seam_edges(mask, (320, 240), bgr) is None
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: FAIL — cannot import `find_seam_edges`.

**Step 3: Add implementation** (ported verbatim; tuning values parameterized)

```python
def _pick_seam_axis(bgr, mask, a0, a1, dark_v_max):
    """Pick which rect axis the seam runs along. a0/a1 = (unit_vec, length)."""
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = (v < dark_v_max) & (mask > 0)
    ys, xs = np.nonzero(dark)
    if xs.size >= 20:
        pts = np.column_stack([xs, ys]).astype(np.float32)
        # the seam is long & thin -> dark pixels spread most along the seam axis
        if (pts @ a0[0]).var() >= (pts @ a1[0]).var():
            return a0
        return a1
    # fallback: clearly-longer axis, else the more-vertical one (near-square)
    if abs(a0[1] - a1[1]) > 0.15 * max(a0[1], a1[1]):
        return a0 if a0[1] >= a1[1] else a1
    return a0 if abs(a0[0][1]) >= abs(a1[0][1]) else a1


def find_seam_edges(
    mask,
    center,
    bgr,
    dark_v_max=SEAM_DARK_V_MAX,
    min_seam_len_px=MIN_SEAM_LEN_PX,
):
    """Top/bottom seam-edge pixels from the box mask, along the seam axis.

    Returns (top_px, bottom_px, angle_deg) -- endpoints at the midpoints of the
    two box edges the seam crosses -- or None if no plausible seam is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    box_pts = cv2.boxPoints(cv2.minAreaRect(contour))
    e0 = box_pts[1] - box_pts[0]
    e1 = box_pts[3] - box_pts[0]
    len0, len1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if len0 < 1e-6 or len1 < 1e-6:
        return None
    a0 = (e0 / len0, len0)
    a1 = (e1 / len1, len1)

    axis, seam_len = _pick_seam_axis(bgr, mask, a0, a1, dark_v_max)
    if seam_len < min_seam_len_px:
        return None

    cx, cy = center
    half = seam_len / 2.0
    top_px = (int(round(cx + axis[0][0] * half)), int(round(cy + axis[0][1] * half)))
    bottom_px = (int(round(cx - axis[0][0] * half)), int(round(cy - axis[0][1] * half)))
    angle_deg = float(np.degrees(np.arctan2(axis[0][1], axis[0][0])))
    return top_px, bottom_px, angle_deg
```

> **Note for implementer:** The original script indexed `axis[0]`/`axis[1]` where
> `axis` was the `(unit_vec, length)` tuple — that was a latent bug (it multiplied
> by the unit-vector array and the scalar length). Here `axis` is the tuple, so use
> `axis[0][0]` / `axis[0][1]` for the unit-vector components. The test above locks
> in the correct vertical-seam behavior.

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_detection.py -v`
Expected: PASS (14 passed).

**Step 5: Commit**

```bash
git add src/models/detection.py tests/test_detection.py
git commit -m "feat: add find_seam_edges and seam-axis picker"
```

---

## Task 7: Config settings dataclass + `validate_config`

**Files:**
- Modify: `src/models/control.py`
- Test: `tests/test_control_config.py`

**Step 1: Write the failing test**

```python
# tests/test_control_config.py
import pytest
from google.protobuf.struct_pb2 import Struct, Value, ListValue
from viam.proto.app.robot import ComponentConfig

from models.control import Control, Settings


def _config(attrs):
    struct = Struct()
    struct.update(attrs)
    return ComponentConfig(attributes=struct)


def test_validate_config_requires_camera_arm_tool():
    with pytest.raises(ValueError, match="camera"):
        Control.validate_config(_config({}))


def test_validate_config_returns_required_deps():
    cfg = _config({"camera": "realsense-cam", "arm": "xarm", "tool_frame": "stylus-tool"})
    required, optional = Control.validate_config(cfg)
    assert set(required) == {"realsense-cam", "xarm"}
    assert optional == []


def test_validate_config_includes_named_motion_service_dep():
    cfg = _config({
        "camera": "realsense-cam", "arm": "xarm", "tool_frame": "stylus-tool",
        "motion_service": "builtin",
    })
    required, _ = Control.validate_config(cfg)
    assert "builtin" in required


def test_settings_from_config_defaults():
    cfg = _config({"camera": "cam", "arm": "arm", "tool_frame": "tool"})
    s = Settings.from_config(cfg)
    assert s.camera_name == "cam"
    assert s.arm_name == "arm"
    assert s.motion_name == "builtin"        # default
    assert s.camera_frame == "cam"           # defaults to camera name
    assert s.world_frame == "world"          # default
    assert s.tool_frame == "tool"
    assert s.min_box_area == 5000            # default tuning
    assert s.plunge_depth_mm == 4
    assert s.hsv_lower == (10, 60, 80)


def test_settings_from_config_overrides_tuning():
    cfg = _config({
        "camera": "cam", "arm": "arm", "tool_frame": "tool",
        "camera_frame": "cam-frame", "world_frame": "map",
        "min_box_area": 1234, "inset_mm": 3, "plunge_depth_mm": 7,
        "hsv_lower": [1, 2, 3], "hsv_upper": [4, 5, 6],
    })
    s = Settings.from_config(cfg)
    assert s.camera_frame == "cam-frame"
    assert s.world_frame == "map"
    assert s.min_box_area == 1234
    assert s.inset_mm == 3
    assert s.plunge_depth_mm == 7
    assert s.hsv_lower == (1, 2, 3)
    assert s.hsv_upper == (4, 5, 6)
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_control_config.py -v`
Expected: FAIL — cannot import `Settings`.

**Step 3: Rewrite `src/models/control.py`**

Replace the file with the following. This adds the `Settings` dataclass, real
`validate_config`, and `reconfigure`; the `do_command` body is a placeholder filled
in Task 9.

```python
from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import MotionClient
from viam.utils import ValueTypes


def _str(config: ComponentConfig, key: str, default: Optional[str] = None) -> Optional[str]:
    fields = config.attributes.fields
    if key in fields and fields[key].string_value:
        return fields[key].string_value
    return default


def _num(config: ComponentConfig, key: str, default):
    fields = config.attributes.fields
    if key in fields and fields[key].HasField("number_value"):
        return type(default)(fields[key].number_value)
    return default


def _triple(config: ComponentConfig, key: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    fields = config.attributes.fields
    if key in fields and fields[key].HasField("list_value"):
        vals = [int(v.number_value) for v in fields[key].list_value.values]
        if len(vals) != 3:
            raise ValueError(f"'{key}' must have exactly 3 numbers")
        return (vals[0], vals[1], vals[2])
    return default


@dataclass
class Settings:
    camera_name: str
    arm_name: str
    tool_frame: str
    motion_name: str
    camera_frame: str
    world_frame: str
    # detection tuning
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    min_box_area: int
    inset_mm: float
    min_seam_len_px: int
    seam_dark_v_max: int
    # motion tuning
    plunge_depth_mm: float
    slice_depth_mm: float
    lift_mm: float
    twist_joint_index: int
    twist_angle_deg: float
    line_tolerance_mm: float

    @classmethod
    def from_config(cls, config: ComponentConfig) -> "Settings":
        camera_name = _str(config, "camera")
        arm_name = _str(config, "arm")
        tool_frame = _str(config, "tool_frame")
        if not camera_name:
            raise ValueError("'camera' is required")
        if not arm_name:
            raise ValueError("'arm' is required")
        if not tool_frame:
            raise ValueError("'tool_frame' is required")
        return cls(
            camera_name=camera_name,
            arm_name=arm_name,
            tool_frame=tool_frame,
            motion_name=_str(config, "motion_service", "builtin"),
            camera_frame=_str(config, "camera_frame", camera_name),
            world_frame=_str(config, "world_frame", "world"),
            hsv_lower=_triple(config, "hsv_lower", (10, 60, 80)),
            hsv_upper=_triple(config, "hsv_upper", (35, 255, 255)),
            min_box_area=_num(config, "min_box_area", 5000),
            inset_mm=_num(config, "inset_mm", 8.0),
            min_seam_len_px=_num(config, "min_seam_len_px", 60),
            seam_dark_v_max=_num(config, "seam_dark_v_max", 80),
            plunge_depth_mm=_num(config, "plunge_depth_mm", 4.0),
            slice_depth_mm=_num(config, "slice_depth_mm", 3.0),
            lift_mm=_num(config, "lift_mm", 100.0),
            twist_joint_index=_num(config, "twist_joint_index", 5),
            twist_angle_deg=_num(config, "twist_angle_deg", 180.0),
            line_tolerance_mm=_num(config, "line_tolerance_mm", 1.0),
        )


class Control(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "box-cutter"), "control")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        settings = Settings.from_config(config)  # raises ValueError on bad config
        required = [settings.camera_name, settings.arm_name, settings.motion_name]
        return required, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self.settings = Settings.from_config(config)
        self.camera = Camera.from_robot_dependencies(dependencies, self.settings.camera_name) \
            if hasattr(Camera, "from_robot_dependencies") else None
        # Dependency resolution is finalized in Task 8; store raw deps for now.
        self._dependencies = dependencies

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`get_status` is not implemented")
        raise NotImplementedError()
```

> **Note:** The `reconfigure` dependency lookup above is a stub — Task 8 replaces it
> with the correct `ResourceName`-based lookup. Keep it compiling for now.

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_control_config.py -v`
Expected: PASS (5 passed).

**Step 5: Run the full suite**

Run: `venv/bin/python -m pytest -v`
Expected: PASS (all detection + config tests).

**Step 6: Commit**

```bash
git add src/models/control.py tests/test_control_config.py
git commit -m "feat: parse and validate box-cutter control config"
```

---

## Task 8: Dependency resolution in `reconfigure`

**Files:**
- Modify: `src/models/control.py`
- Test: `tests/test_control_config.py`

**Goal:** Resolve `camera`/`arm`/`motion` clients from the dependency mapping using
their `ResourceName`, following the standard module pattern.

**Step 1: Add a failing test**

```python
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.services.motion import MotionClient


class _FakeCam(Camera):
    async def get_images(self, *a, **k): ...
    async def get_image(self, *a, **k): ...
    async def get_point_cloud(self, *a, **k): ...
    async def get_properties(self, *a, **k): ...
    async def do_command(self, *a, **k): ...


def test_reconfigure_resolves_dependencies():
    cfg = _config({"camera": "cam", "arm": "arm", "tool_frame": "tool"})
    cam = _FakeCam("cam")
    deps = {
        Camera.get_resource_name("cam"): cam,
    }
    ctrl = Control("control")
    # Only camera provided; arm/motion resolution is exercised on the live machine,
    # but camera lookup must succeed here.
    ctrl.settings = Settings.from_config(cfg)
    ctrl.camera = Control._resolve(deps, Camera.get_resource_name("cam"))
    assert ctrl.camera is cam
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_control_config.py::test_reconfigure_resolves_dependencies -v`
Expected: FAIL — `Control` has no `_resolve`.

**Step 3: Replace the `reconfigure` method and add `_resolve`**

```python
    @staticmethod
    def _resolve(dependencies: Mapping[ResourceName, ResourceBase], name: ResourceName):
        resource = dependencies.get(name)
        if resource is None:
            raise ValueError(f"missing required dependency: {name.name}")
        return resource

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        settings = Settings.from_config(config)
        self.settings = settings
        self.camera = self._resolve(
            dependencies, Camera.get_resource_name(settings.camera_name)
        )
        self.arm = self._resolve(
            dependencies, Arm.get_resource_name(settings.arm_name)
        )
        self.motion = self._resolve(
            dependencies, MotionClient.get_resource_name(settings.motion_name)
        )
```

Remove the placeholder `self.camera = Camera.from_robot_dependencies(...)` block and
the `self._dependencies = dependencies` line from Task 7.

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_control_config.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/control.py tests/test_control_config.py
git commit -m "feat: resolve camera/arm/motion dependencies in reconfigure"
```

---

## Task 9: `do_command` dispatch + `find_center`

**Files:**
- Modify: `src/models/control.py`
- Test: `tests/test_control_dispatch.py`

**Design of `find_center`:**
1. `images, _ = await self.camera.get_images()`; `properties = await self.camera.get_properties()`; `intr = properties.intrinsic_parameters`.
2. `color_bgr, depth_np = decode_color_and_depth(images)`; raise `ValueError` if `depth_np.shape[:2] != color_bgr.shape[:2]`.
3. `u, v, mask = detect_box_center(color_bgr, hsv_lower, hsv_upper, min_box_area)`; if `u is None`: return `{"found": False, "reason": "no box-colored region found"}`.
4. `z = sample_depth_in_mask(depth_np, mask)`; if `None`: return `{"found": False, "reason": "no valid depth in box mask"}`.
5. `cx, cy, cz = deproject(u, v, z, intr)`; `world = await self._to_frame((cx, cy, cz), self.settings.world_frame)`.
6. `seam = find_seam_edges(mask, (u, v), color_bgr, seam_dark_v_max, min_seam_len_px)`. If present, deproject each endpoint at depth `z`, transform to world, then `inset_endpoints(...)`.
7. Build a JSON-native dict (all `float`/`int`/`list`).

The frame-transform helper (added here):

```python
    async def _to_frame(self, point_xyz, dest_frame):
        """Transform a camera-frame point (mm) into dest_frame via the motion service.

        Replaces RobotClient.transform_pose (unavailable in a module) by injecting
        the point as a temporary frame parented to the camera frame, then asking the
        motion service for its pose in dest_frame.
        """
        x, y, z = point_xyz
        transform = Transform(
            reference_frame="box_point",
            pose_in_observer_frame=PoseInFrame(
                reference_frame=self.settings.camera_frame,
                pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0),
            ),
        )
        return await self.motion.get_pose(
            component_name="box_point",
            destination_frame=dest_frame,
            supplemental_transforms=[transform],
        )
```

**Step 1: Write the failing test** (fakes for camera + motion so no hardware needed)

```python
# tests/test_control_dispatch.py
import numpy as np
import cv2
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame
from viam.media.video import CameraMimeType

from models.control import Control, Settings


def _config(attrs):
    s = Struct(); s.update(attrs)
    return ComponentConfig(attributes=s)


class _Intr:
    focal_x_px = 600.0
    focal_y_px = 600.0
    center_x_px = 320.0
    center_y_px = 240.0


class _Props:
    intrinsic_parameters = _Intr()


class _NamedImage:
    def __init__(self, mime_type, data=b"", depth=None):
        self.mime_type = mime_type
        self.data = data
        self._depth = depth
    def bytes_to_depth_array(self):
        return self._depth


class _FakeCamera:
    def __init__(self, images):
        self._images = images
    async def get_images(self):
        return self._images, None
    async def get_properties(self):
        return _Props()


class _FakeMotion:
    async def get_pose(self, component_name, destination_frame, supplemental_transforms=None, **kw):
        # Echo the injected camera-frame point back as if it were world coords.
        t = supplemental_transforms[0].pose_in_observer_frame.pose
        return PoseInFrame(
            reference_frame=destination_frame,
            pose=Pose(x=t.x, y=t.y, z=t.z, o_x=0, o_y=0, o_z=-1, theta=0),
        )


def _images_with_box():
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    bgr[140:340, 240:400] = (60, 140, 200)   # tan box, center ~ (320, 240)
    bgr[140:340, 316:324] = (10, 10, 10)      # dark seam
    ok, buf = cv2.imencode(".jpg", bgr)
    depth = np.full((480, 640), 700, dtype=np.uint16)
    return [
        _NamedImage(CameraMimeType.JPEG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ]


def _make_control():
    ctrl = Control("control")
    ctrl.settings = Settings.from_config(
        _config({"camera": "cam", "arm": "arm", "tool_frame": "tool"})
    )
    ctrl.camera = _FakeCamera(_images_with_box())
    ctrl.motion = _FakeMotion()
    ctrl.arm = None
    return ctrl


@pytest.mark.asyncio
async def test_find_center_returns_world_pose():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "find_center"})
    assert out["found"] is True
    assert abs(out["u"] - 320) <= 5
    assert abs(out["v"] - 240) <= 5
    assert out["depth_mm"] == pytest.approx(700.0)
    # center pixel ~ principal point -> deprojected x,y ~ 0
    assert abs(out["world_pose"]["x"]) < 5
    assert abs(out["world_pose"]["y"]) < 5
    assert out["world_pose"]["z"] == pytest.approx(700.0, abs=1)
    assert "cut_endpoints_world" in out


@pytest.mark.asyncio
async def test_find_center_reports_not_found_on_blank_frame():
    ctrl = _make_control()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    depth = np.full((480, 640), 700, dtype=np.uint16)
    ctrl.camera = _FakeCamera([
        _NamedImage(CameraMimeType.JPEG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ])
    out = await ctrl.do_command({"command": "find_center"})
    assert out["found"] is False
    assert "reason" in out


@pytest.mark.asyncio
async def test_do_command_unknown_raises():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="unknown command"):
        await ctrl.do_command({"command": "bogus"})


@pytest.mark.asyncio
async def test_do_command_missing_command_raises():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="command"):
        await ctrl.do_command({})
```

Add `pytest-asyncio` to `requirements-dev.txt` and a marker config. Update
`pytest.ini`:

```ini
[pytest]
pythonpath = src
testpaths = tests
asyncio_mode = auto
```

And append `pytest-asyncio` to `requirements-dev.txt`, then
`venv/bin/python -m pip install -r requirements-dev.txt`.

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_control_dispatch.py -v`
Expected: FAIL — `do_command` raises `NotImplementedError`.

**Step 3: Implement dispatch + `find_center` + `_to_frame`**

Add imports at the top of `control.py`:

```python
from viam.proto.common import Pose, PoseInFrame, Transform

from models.detection import (
    decode_color_and_depth,
    deproject,
    detect_box_center,
    find_seam_edges,
    inset_endpoints,
    sample_depth_in_mask,
)
```

Replace `do_command` and add the action methods:

```python
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        name = command.get("command")
        if not name:
            raise ValueError("do_command requires a 'command' key")
        if name == "find_center":
            return await self.find_center()
        if name == "move_to_center":
            return await self.move_to_center()
        if name == "full_cut":
            return await self.full_cut()
        raise ValueError(f"unknown command: {name!r}")

    async def find_center(self) -> Mapping[str, ValueTypes]:
        s = self.settings
        images, _ = await self.camera.get_images()
        properties = await self.camera.get_properties()
        intr = properties.intrinsic_parameters

        color_bgr, depth_np = decode_color_and_depth(images)
        if depth_np.shape[:2] != color_bgr.shape[:2]:
            raise ValueError(
                f"depth {depth_np.shape[:2]} not aligned to color "
                f"{color_bgr.shape[:2]}; cannot deproject with color intrinsics"
            )

        u, v, mask = detect_box_center(color_bgr, s.hsv_lower, s.hsv_upper, s.min_box_area)
        if u is None:
            return {"found": False, "reason": "no box-colored region found"}

        z = sample_depth_in_mask(depth_np, mask)
        if z is None:
            return {"found": False, "reason": "no valid depth in box mask"}

        cx, cy, cz = deproject(u, v, z, intr)
        world = await self._to_frame((cx, cy, cz), s.world_frame)

        result = {
            "found": True,
            "u": int(u),
            "v": int(v),
            "depth_mm": float(z),
            "camera_frame_xyz": {"x": float(cx), "y": float(cy), "z": float(cz)},
            "world_pose": _pose_to_dict(world.pose),
        }

        seam = find_seam_edges(mask, (u, v), color_bgr, s.seam_dark_v_max, s.min_seam_len_px)
        if seam is not None:
            top_px, bottom_px, angle_deg = seam
            top_w = await self._endpoint_world(top_px, z, intr)
            bottom_w = await self._endpoint_world(bottom_px, z, intr)
            top_inset, bottom_inset = inset_endpoints(top_w, bottom_w, s.inset_mm)
            result["seam"] = {
                "top_px": [int(top_px[0]), int(top_px[1])],
                "bottom_px": [int(bottom_px[0]), int(bottom_px[1])],
                "angle_deg": float(angle_deg),
            }
            result["cut_endpoints_world"] = {
                "top": [float(c) for c in top_inset],
                "bottom": [float(c) for c in bottom_inset],
            }
        return result

    async def _endpoint_world(self, px, z, intr):
        ex, ey, ez = deproject(px[0], px[1], z, intr)
        pif = await self._to_frame((ex, ey, ez), self.settings.world_frame)
        return (pif.pose.x, pif.pose.y, pif.pose.z)

    async def _to_frame(self, point_xyz, dest_frame):
        """Transform a camera-frame point (mm) into dest_frame via the motion service."""
        x, y, z = point_xyz
        transform = Transform(
            reference_frame="box_point",
            pose_in_observer_frame=PoseInFrame(
                reference_frame=self.settings.camera_frame,
                pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0),
            ),
        )
        return await self.motion.get_pose(
            component_name="box_point",
            destination_frame=dest_frame,
            supplemental_transforms=[transform],
        )
```

Add a module-level helper near the top of `control.py`:

```python
def _pose_to_dict(pose) -> dict:
    return {
        "x": float(pose.x), "y": float(pose.y), "z": float(pose.z),
        "o_x": float(pose.o_x), "o_y": float(pose.o_y), "o_z": float(pose.o_z),
        "theta": float(pose.theta),
    }
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_control_dispatch.py -v`
Expected: PASS (4 passed).

**Step 5: Commit**

```bash
git add src/models/control.py tests/test_control_dispatch.py requirements-dev.txt pytest.ini
git commit -m "feat: implement do_command dispatch and find_center action"
```

---

## Task 10: `move_to_center`

**Files:**
- Modify: `src/models/control.py`
- Test: `tests/test_control_dispatch.py`

**Step 1: Add a failing test** (fake motion records the move goal)

```python
class _RecordingMotion(_FakeMotion):
    def __init__(self):
        self.moved = None
    async def move(self, component_name, destination, **kw):
        self.moved = (component_name, destination)
        return True


@pytest.mark.asyncio
async def test_move_to_center_commands_move_to_world_center():
    ctrl = _make_control()
    ctrl.motion = _RecordingMotion()
    out = await ctrl.do_command({"command": "move_to_center"})
    assert out["found"] is True
    assert out["moved"] is True
    comp, dest = ctrl.motion.moved
    assert comp == "tool"                      # tool_frame
    assert dest.reference_frame == "world"
    # z pulled down by plunge_depth (default 4) from ~700
    assert dest.pose.z == pytest.approx(700.0 - 4.0, abs=1)
    assert dest.pose.o_z == -1


@pytest.mark.asyncio
async def test_move_to_center_skips_move_when_not_found():
    ctrl = _make_control()
    ctrl.motion = _RecordingMotion()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    depth = np.full((480, 640), 700, dtype=np.uint16)
    ctrl.camera = _FakeCamera([
        _NamedImage(CameraMimeType.JPEG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ])
    out = await ctrl.do_command({"command": "move_to_center"})
    assert out["found"] is False
    assert ctrl.motion.moved is None
```

> **Note:** `_RecordingMotion` extends `_FakeMotion`, so `get_pose` still echoes the
> injected point — meaning the "world" center z equals the input depth (700). This
> lets the test assert the plunge offset deterministically.

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_control_dispatch.py -k move_to_center -v`
Expected: FAIL — `move_to_center` not implemented / returns nothing useful.

**Step 3: Implement `move_to_center`**

```python
    async def move_to_center(self) -> Mapping[str, ValueTypes]:
        result = await self.find_center()
        if not result.get("found"):
            return result
        s = self.settings
        w = result["world_pose"]
        goal = PoseInFrame(
            reference_frame=s.world_frame,
            pose=Pose(
                x=w["x"], y=w["y"], z=w["z"] - s.plunge_depth_mm,
                o_x=0, o_y=0, o_z=-1, theta=0,
            ),
        )
        await self.motion.move(component_name=s.tool_frame, destination=goal)
        result["moved"] = True
        return result
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_control_dispatch.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/control.py tests/test_control_dispatch.py
git commit -m "feat: implement move_to_center action"
```

---

## Task 11: `full_cut`

**Files:**
- Modify: `src/models/control.py`
- Test: `tests/test_control_dispatch.py`

**Behavior** (ported from the commented block in the script, parameterized by tuning
attributes). Uses `self.arm` for joint twist and `self.motion` for moves. Because the
sequence depends on live IK/frame-system results, the test verifies **orchestration**
(the right calls in the right order), not physical correctness.

**Step 1: Add a failing test**

```python
from viam.proto.common import Pose as _Pose


class _FakeJointPositions:
    def __init__(self, values):
        self.values = list(values)


class _FakeArm:
    def __init__(self):
        self.joint_history = []
    async def get_joint_positions(self, **kw):
        return _FakeJointPositions([0, 0, 0, 0, 0, 0])
    async def move_to_joint_positions(self, positions, **kw):
        self.joint_history.append(list(positions.values))


class _FullCutMotion(_RecordingMotion):
    def __init__(self):
        super().__init__()
        self.moves = []
        self._home = PoseInFrame(
            reference_frame="world",
            pose=_Pose(x=1, y=2, z=3, o_x=0, o_y=0, o_z=-1, theta=0),
        )
    async def move(self, component_name, destination, **kw):
        self.moves.append((component_name, destination, kw.get("constraints")))
        return True
    async def get_pose(self, component_name, destination_frame, supplemental_transforms=None, **kw):
        if supplemental_transforms is None:
            # home-pose query for the arm
            return self._home
        return await super().get_pose(component_name, destination_frame, supplemental_transforms)


@pytest.mark.asyncio
async def test_full_cut_runs_full_sequence():
    ctrl = _make_control()
    ctrl.motion = _FullCutMotion()
    ctrl.arm = _FakeArm()
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["completed"] is True
    # twist to angle then back to 0
    assert ctrl.arm.joint_history[0][ctrl.settings.twist_joint_index] == ctrl.settings.twist_angle_deg
    assert ctrl.arm.joint_history[-1][ctrl.settings.twist_joint_index] == 0
    # at least: move_to_center + plunge + lift + slice + lift + home
    assert len(ctrl.motion.moves) >= 5
    # the slice move carries a linear constraint
    assert any(c is not None for (_, _, c) in ctrl.motion.moves)


@pytest.mark.asyncio
async def test_full_cut_aborts_when_no_box():
    ctrl = _make_control()
    ctrl.motion = _FullCutMotion()
    ctrl.arm = _FakeArm()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    depth = np.full((480, 640), 700, dtype=np.uint16)
    ctrl.camera = _FakeCamera([
        _NamedImage(CameraMimeType.JPEG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ])
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["found"] is False
    assert ctrl.arm.joint_history == []
```

**Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_control_dispatch.py -k full_cut -v`
Expected: FAIL — `full_cut` not implemented.

**Step 3: Implement `full_cut`**

Add the import for the linear constraint at the top:

```python
from viam.services.motion import Constraints
from viam.proto.service.motion import LinearConstraint
```

Then:

```python
    async def full_cut(self) -> Mapping[str, ValueTypes]:
        s = self.settings
        # Capture the arm's home pose before moving so we can return to it.
        home = await self.motion.get_pose(
            component_name=s.arm_name, destination_frame=s.world_frame
        )

        result = await self.move_to_center()
        if not result.get("found"):
            return result
        if "cut_endpoints_world" not in result:
            result["completed"] = False
            result["reason"] = "no seam/cut endpoints detected; only moved to center"
            return result

        steps = ["move_to_center"]
        top = result["cut_endpoints_world"]["top"]
        bottom = result["cut_endpoints_world"]["bottom"]
        w = result["world_pose"]

        def down_pose(x, y, z):
            return PoseInFrame(
                reference_frame=s.world_frame,
                pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0),
            )

        # 1) plunge at the bottom inset endpoint
        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(bottom[0], bottom[1], bottom[2] - s.slice_depth_mm),
        )
        steps.append("plunge_bottom")

        # 2) lift
        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(bottom[0], bottom[1], bottom[2] - s.slice_depth_mm + s.lift_mm),
        )
        steps.append("lift")

        # 3) twist the wrist joint
        joints = await self.arm.get_joint_positions()
        joints.values[s.twist_joint_index] = s.twist_angle_deg
        await self.arm.move_to_joint_positions(joints)
        steps.append("twist")

        # 4) linear-constrained slice to the top inset endpoint
        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(top[0], top[1], bottom[2] - s.slice_depth_mm),
            constraints=Constraints(
                linear_constraint=[LinearConstraint(line_tolerance_mm=s.line_tolerance_mm)]
            ),
        )
        steps.append("slice_top")

        # 5) lift
        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(top[0], top[1], bottom[2] - s.slice_depth_mm + s.lift_mm),
        )
        steps.append("lift")

        # 6) untwist
        joints = await self.arm.get_joint_positions()
        joints.values[s.twist_joint_index] = 0
        await self.arm.move_to_joint_positions(joints)
        steps.append("untwist")

        # 7) return the arm to its captured home pose
        await self.motion.move(component_name=s.arm_name, destination=home)
        steps.append("home")

        result["completed"] = True
        result["steps"] = steps
        return result
```

**Step 4: Run to verify pass**

Run: `venv/bin/python -m pytest tests/test_control_dispatch.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/control.py tests/test_control_dispatch.py
git commit -m "feat: implement full_cut sequence"
```

---

## Task 12: Full suite + module import smoke test

**Files:** none (verification only)

**Step 1: Run the entire test suite**

Run: `venv/bin/python -m pytest -v`
Expected: all tests pass.

**Step 2: Verify the module imports cleanly (registration path)**

Run:
```bash
venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from models.control import Control; print(Control.MODEL)"
```
Expected: prints `viam-labs:box-cutter:control` with no import errors.

**Step 3: Verify `main.py` entrypoint still parses**

Run:
```bash
venv/bin/python -c "import ast; ast.parse(open('src/main.py').read()); print('main.py ok')"
```
Expected: `main.py ok`.

**Step 4: Commit** (if any incidental fixes were needed)

```bash
git add -A
git commit -m "test: verify full suite and module import"
```

---

## Task 13: Documentation

**Files:**
- Modify: `viam-labs_box-cutter_control.md` (the generated model doc) and/or `README.md`

**Step 1:** Document the three `do_command` actions and every config attribute
(required: `camera`, `arm`, `tool_frame`; optional: `motion_service`, `camera_frame`,
`world_frame`, and the detection/motion tuning keys with their defaults). Include an
example config and an example `do_command` call:

```json
{ "command": "find_center" }
```

and the shape of the returned payload (from the design doc).

**Step 2: Commit**

```bash
git add viam-labs_box-cutter_control.md README.md
git commit -m "docs: document control do_command actions and config"
```

---

## Manual / hardware verification (out of automated scope)

After deploying to a machine with the RealSense camera, xArm, and motion service:

1. `find_center` — confirm returned `world_pose` matches the physical box center;
   compare against the old script's printed values.
2. `move_to_center` — confirm the stylus descends to the box center.
3. `full_cut` — dry-run with the arm clear, then with a box, watching each step.

Use @superpowers:verification-before-completion before claiming the feature done.
