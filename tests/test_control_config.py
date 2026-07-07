import pytest
from google.protobuf.struct_pb2 import Struct
from viam.components.camera import Camera
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
    assert set(required) == {"realsense-cam", "xarm", "builtin"}
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
    assert s.motion_name == "builtin"
    assert s.camera_frame == "cam"
    assert s.world_frame == "world"
    assert s.tool_frame == "tool"
    assert s.min_box_area == 5000
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


class _FakeCam(Camera):
    async def get_images(self, *a, **k): ...
    async def get_image(self, *a, **k): ...
    async def get_point_cloud(self, *a, **k): ...
    async def get_properties(self, *a, **k): ...
    async def do_command(self, *a, **k): ...


def test_resolve_returns_matching_dependency():
    cam = _FakeCam("cam")
    deps = {Camera.get_resource_name("cam"): cam}
    assert Control._resolve(deps, Camera.get_resource_name("cam")) is cam


def test_resolve_raises_on_missing_dependency():
    with pytest.raises(ValueError, match="missing required dependency"):
        Control._resolve({}, Camera.get_resource_name("cam"))
