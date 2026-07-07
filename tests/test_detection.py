import cv2
import numpy as np
import pytest

from viam.media.video import CameraMimeType

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
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(0.0)
    assert z == 1000.0


def test_deproject_uses_independent_y_and_focal_y():
    intr = _Intr(fx=600.0, fy=400.0, cx=320.0, cy=240.0)
    u, v, z = 320 + 60, 240 + 40, 1000.0
    x, y, z_out = deproject(u, v, z, intr)
    assert x == pytest.approx((u - 320.0) * z / 600.0)
    assert y == pytest.approx((v - 240.0) * z / 400.0)
    assert z_out == 1000.0


from models.detection import inset_endpoints


def test_inset_endpoints_pulls_inward():
    top, bottom = inset_endpoints((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), inset_mm=10)
    assert top == pytest.approx((10.0, 0.0, 0.0))
    assert bottom == pytest.approx((90.0, 0.0, 0.0))


def test_inset_endpoints_collapses_when_too_short():
    top, bottom = inset_endpoints((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), inset_mm=8)
    assert top == pytest.approx((5.0, 0.0, 0.0))
    assert bottom == pytest.approx((5.0, 0.0, 0.0))


from models.detection import sample_depth_in_mask


def test_sample_depth_median_ignores_zeros():
    depth = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
    mask = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    assert sample_depth_in_mask(depth, mask) == pytest.approx(2000.0)


def test_sample_depth_returns_none_when_all_invalid():
    depth = np.zeros((2, 2), dtype=np.uint16)
    mask = np.full((2, 2), 255, dtype=np.uint8)
    assert sample_depth_in_mask(depth, mask) is None


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
    bgr[:] = (0, 0, 200)
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


from models.detection import detect_box_center


def _tan_box_image(w=640, h=480, box=(220, 160, 200, 160)):
    """Neutral-gray background with one solid tan (HSV ~20,150,200) rectangle."""
    bgr = np.full((h, w, 3), 128, dtype=np.uint8)
    x, y, bw, bh = box
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
    bgr, _ = _tan_box_image(box=(10, 10, 20, 20))
    cx, cy, mask = detect_box_center(bgr)
    assert cx is None and cy is None and mask is None


def test_detect_box_center_none_when_no_box_color():
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    cx, cy, mask = detect_box_center(bgr)
    assert cx is None and cy is None and mask is None


from models.detection import find_seam_edges


def test_find_seam_edges_returns_endpoints_along_dark_seam():
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
    assert abs(top_px[0] - center[0]) <= 10
    assert abs(bottom_px[0] - center[0]) <= 10
    assert abs(top_px[1] - bottom_px[1]) > 50


def test_find_seam_edges_none_on_empty_mask():
    mask = np.zeros((480, 640), dtype=np.uint8)
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    assert find_seam_edges(mask, (320, 240), bgr) is None


def test_find_seam_edges_falls_back_to_longer_axis_without_dark_seam():
    h, w = 480, 640
    bgr = np.full((h, w, 3), 128, dtype=np.uint8)
    # Plain tan rectangle, no dark seam: 240 wide x 120 tall (clearly non-square).
    x0, y0, bw, bh = 200, 180, 240, 120
    bgr[y0:y0 + bh, x0:x0 + bw] = (60, 140, 200)
    _, _, mask = detect_box_center(bgr)
    assert mask is not None
    center = (x0 + bw // 2, y0 + bh // 2)
    result = find_seam_edges(mask, center, bgr)
    assert result is not None
    top_px, bottom_px, angle_deg = result
    # No dark pixels -> the length fallback picks the clearly-longer (horizontal) axis.
    dx = abs(top_px[0] - bottom_px[0])
    dy = abs(top_px[1] - bottom_px[1])
    assert dx > dy
