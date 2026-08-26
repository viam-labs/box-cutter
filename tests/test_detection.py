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


from models.detection import (
    find_vertical_seam_line,
    invert_jacobian,
    pixel_error_to_delta_mm,
)


def _frame_with_vertical_lines(*x_positions, height=(100, 400)):
    """Light background with dark vertical bars at the given x columns."""
    bgr = np.full((480, 640, 3), 200, dtype=np.uint8)
    top, bottom = height
    for x in x_positions:
        bgr[top:bottom, x - 1:x + 2] = 20
    return bgr


def test_find_vertical_seam_line_locates_bar_near_blade():
    bgr = _frame_with_vertical_lines(350)
    found = find_vertical_seam_line(bgr, blade_x_px=339)
    assert found is not None
    center_x, center_y, segment = found
    assert center_x == pytest.approx(350, abs=3)
    assert 100 <= center_y <= 400
    assert len(segment) == 4


def test_find_vertical_seam_line_returns_none_on_blank_frame():
    bgr = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert find_vertical_seam_line(bgr, blade_x_px=339) is None


def test_find_vertical_seam_line_ignores_line_outside_search_radius():
    bgr = _frame_with_vertical_lines(500)
    assert find_vertical_seam_line(bgr, blade_x_px=339, search_radius_px=40) is None


def test_find_vertical_seam_line_ignores_horizontal_line():
    bgr = np.full((480, 640, 3), 200, dtype=np.uint8)
    bgr[238:242, 200:500] = 20
    assert find_vertical_seam_line(bgr, blade_x_px=339) is None


def test_find_vertical_seam_line_picks_nearest_candidate():
    bgr = _frame_with_vertical_lines(345, 370)
    found = find_vertical_seam_line(bgr, blade_x_px=339, search_radius_px=40)
    assert found is not None
    assert found[0] == pytest.approx(345, abs=3)


def test_find_vertical_seam_line_accepts_perfectly_aligned_line():
    # The original script's filter was `0 < distance < 40`, which rejected the
    # exactly-centered case -- i.e. the converged one.
    bgr = _frame_with_vertical_lines(339)
    found = find_vertical_seam_line(bgr, blade_x_px=339)
    assert found is not None
    assert found[0] == pytest.approx(339, abs=3)


def test_invert_jacobian_round_trips():
    j = ((-1.0, 0.1), (0.2, -1.0))
    inv = invert_jacobian(j)
    # inv @ j should be the identity
    for i in range(2):
        for k in range(2):
            entry = sum(inv[i][m] * j[m][k] for m in range(2))
            assert entry == pytest.approx(1.0 if i == k else 0.0, abs=1e-9)


def test_invert_jacobian_rejects_singular():
    with pytest.raises(ValueError, match="singular"):
        invert_jacobian(((1.0, 2.0), (2.0, 4.0)))


def test_pixel_error_to_delta_mm_steps_against_the_error():
    inv = invert_jacobian(((-1.0, 0.0), (0.0, -1.0)))
    dx, dy = pixel_error_to_delta_mm((10.0, 0.0), inv, gain=0.2)
    # inv(J) = -I, so the step is +gain*error: a blade left of the seam moves right.
    assert dx == pytest.approx(2.0)
    assert dy == pytest.approx(0.0)


def test_pixel_error_to_delta_mm_scales_with_gain():
    inv = invert_jacobian(((-1.0, 0.1), (0.2, -1.0)))
    small = pixel_error_to_delta_mm((10.0, 0.0), inv, gain=0.05)
    large = pixel_error_to_delta_mm((10.0, 0.0), inv, gain=0.20)
    assert large[0] == pytest.approx(small[0] * 4)
    assert large[1] == pytest.approx(small[1] * 4)


def test_pixel_error_to_delta_mm_zero_error_is_zero_step():
    inv = invert_jacobian(((-1.0, 0.1), (0.2, -1.0)))
    assert pixel_error_to_delta_mm((0.0, 0.0), inv, gain=0.2) == (0.0, 0.0)
