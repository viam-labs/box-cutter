import pytest
from google.protobuf.struct_pb2 import Struct
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.app.robot import ComponentConfig
from viam.services.motion import MotionClient

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
    assert s.blade_frame == ""
    assert s.min_box_area == 5000
    assert s.center_standoff_mm == 20
    assert s.hsv_lower == (10, 60, 80)


def test_settings_defaults_carry_the_measured_ground_truth():
    s = Settings.from_config(_config({"camera": "c", "arm": "a", "tool_frame": "t"}))
    assert s.stopper_y_mm == -450.0
    assert s.knife_tip_to_table_mm == 490.0
    assert s.base_plate_height_mm == 20.0
    assert s.home_xyz == (-4.0, -551.0, 470.0)


def test_settings_defaults_carry_the_servo_tuning():
    s = Settings.from_config(_config({"camera": "c", "arm": "a", "tool_frame": "t"}))
    assert s.blade_x_px == 339.0
    assert s.converge_tolerance_px == 2.25
    assert s.servo_jacobian == (-1.0, 0.1, 0.2, -1.0)
    assert (s.top_seam_gain, s.far_seam_gain, s.close_seam_gain) == (0.2, 0.05, 0.09)


def test_settings_overrides_ground_truth_and_home():
    cfg = _config({
        "camera": "c", "arm": "a", "tool_frame": "t",
        "stopper_y_mm": -500, "knife_tip_to_table_mm": 505,
        "base_plate_height_mm": 25, "home_xyz": [0, -600, 500],
    })
    s = Settings.from_config(cfg)
    assert s.stopper_y_mm == -500
    assert s.knife_tip_to_table_mm == 505
    assert s.base_plate_height_mm == 25
    assert s.home_xyz == (0.0, -600.0, 500.0)


def test_settings_ground_truth_stays_float_when_configured_whole():
    # `_num` coerces to the default's type, so an integer-looking override of a
    # float attribute must not silently become an int.
    cfg = _config({"camera": "c", "arm": "a", "tool_frame": "t", "stopper_y_mm": -400})
    s = Settings.from_config(cfg)
    assert isinstance(s.stopper_y_mm, float)


def test_settings_rejects_wrong_length_home_xyz():
    cfg = _config({"camera": "c", "arm": "a", "tool_frame": "t", "home_xyz": [1, 2]})
    with pytest.raises(ValueError, match="exactly 3"):
        Settings.from_config(cfg)


def test_settings_rejects_wrong_length_jacobian():
    cfg = _config({
        "camera": "c", "arm": "a", "tool_frame": "t", "servo_jacobian": [1, 2, 3],
    })
    with pytest.raises(ValueError, match="exactly 4"):
        Settings.from_config(cfg)


def test_settings_rejects_empty_top_seam_chunks():
    cfg = _config({
        "camera": "c", "arm": "a", "tool_frame": "t", "top_seam_chunks": [],
    })
    with pytest.raises(ValueError, match="must not be empty"):
        Settings.from_config(cfg)


def test_top_seam_span_is_the_sum_of_its_chunks():
    s = Settings.from_config(_config({"camera": "c", "arm": "a", "tool_frame": "t"}))
    assert s.top_seam_chunks == (0.2, 0.2, 0.25)
    assert s.top_seam_span_fraction == pytest.approx(0.65)

    cfg = _config({
        "camera": "c", "arm": "a", "tool_frame": "t", "top_seam_chunks": [0.5, 0.25],
    })
    assert Settings.from_config(cfg).top_seam_span_fraction == pytest.approx(0.75)


def test_gain_for_selects_the_per_seam_gain():
    s = Settings.from_config(_config({"camera": "c", "arm": "a", "tool_frame": "t"}))
    assert s.gain_for("top") == 0.2
    assert s.gain_for("far") == 0.05
    assert s.gain_for("close") == 0.09


def test_settings_from_config_overrides_tuning():
    cfg = _config({
        "camera": "cam", "arm": "arm", "tool_frame": "tool",
        "camera_frame": "cam-frame", "world_frame": "map",
        "min_box_area": 1234, "inset_mm": 3, "center_standoff_mm": 7,
        "hsv_lower": [1, 2, 3], "hsv_upper": [4, 5, 6],
    })
    s = Settings.from_config(cfg)
    assert s.camera_frame == "cam-frame"
    assert s.world_frame == "map"
    assert s.min_box_area == 1234
    assert s.inset_mm == 3
    assert s.center_standoff_mm == 7
    assert s.hsv_lower == (1, 2, 3)
    assert s.hsv_upper == (4, 5, 6)


def test_settings_rejects_wrong_length_hsv():
    cfg = _config({"camera": "c", "arm": "a", "tool_frame": "t", "hsv_lower": [1, 2]})
    with pytest.raises(ValueError, match="exactly 3"):
        Settings.from_config(cfg)


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


def test_reconfigure_wires_up_resolved_dependencies():
    ctrl = Control.__new__(Control)
    cam_obj, arm_obj, motion_obj = object(), object(), object()
    deps = {
        Camera.get_resource_name("cam"): cam_obj,
        Arm.get_resource_name("arm"): arm_obj,
        MotionClient.get_resource_name("builtin"): motion_obj,
    }
    ctrl.reconfigure(
        _config({"camera": "cam", "arm": "arm", "tool_frame": "tool"}), deps
    )
    assert ctrl.camera is cam_obj
    assert ctrl.arm is arm_obj
    assert ctrl.motion is motion_obj
    assert ctrl.settings.tool_frame == "tool"


def test_reconfigure_clears_per_box_state():
    ctrl = Control.__new__(Control)
    deps = {
        Camera.get_resource_name("cam"): object(),
        Arm.get_resource_name("arm"): object(),
        MotionClient.get_resource_name("builtin"): object(),
    }
    cfg = _config({"camera": "cam", "arm": "arm", "tool_frame": "tool"})
    ctrl.reconfigure(cfg, deps)
    ctrl._box_override = {"depth_mm": 1}
    ctrl._box_frame = object()
    ctrl.reconfigure(cfg, deps)
    assert ctrl._box_override is None
    assert ctrl._box_frame is None


def test_reconfigure_raises_on_missing_dependency():
    ctrl = Control.__new__(Control)
    deps = {
        Camera.get_resource_name("cam"): object(),
        MotionClient.get_resource_name("builtin"): object(),
    }
    with pytest.raises(ValueError, match="missing required dependency"):
        ctrl.reconfigure(
            _config({"camera": "cam", "arm": "arm", "tool_frame": "tool"}), deps
        )
