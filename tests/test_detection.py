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
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(0.0)
    assert z == 1000.0
