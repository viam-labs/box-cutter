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
