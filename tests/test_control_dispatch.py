import asyncio
import numpy as np
import cv2
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame
from viam.media.video import CameraMimeType

from models.control import (
    BOX_PRESETS,
    CLOSE_SEAM_RETRACT_MM,
    CUT_SIGN,
    FAR_SEAM_APPROACH_LATERAL_MM,
    SIDE_SEAM_THETA_DEG,
    Control,
    Settings,
    SEAM_CLOSE,
    SEAM_FAR,
    SEAM_TOP,
)

# find_center is set_box-only until automatic detection returns; these tests
# describe the detection path and come back with it.
DETECTION_DISABLED = pytest.mark.skip(
    reason="automatic detection is disabled; find_center is set_box-only"
)

BOX_1 = {"depth_mm": 535, "u": 360, "v": 223, "flap_width_mm": 120, "box_height_mm": 280}


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


class _FakeRobotClient:
    """Stands in for RobotClient.transform_pose.

    `world` passes the camera-frame point through unchanged; the tool frame is
    offset so tests can tell the two apart.
    """
    TOOL_OFFSET = (1.0, 2.0, 3.0)

    def __init__(self):
        self.requests = []

    async def transform_pose(self, pose_in_frame, dest_frame):
        self.requests.append((pose_in_frame, dest_frame))
        p = pose_in_frame.pose
        if dest_frame == "world":
            x, y, z = p.x, p.y, p.z
        else:
            dx, dy, dz = self.TOOL_OFFSET
            x, y, z = p.x + dx, p.y + dy, p.z + dz
        return PoseInFrame(
            reference_frame=dest_frame,
            pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0),
        )

    def close(self):
        pass


class _RecordingMotion:
    """Records every move; reports the tool parked at world x = `tool_x`."""
    def __init__(self, tool_x=0.0):
        self.moves = []
        self.tool_x = tool_x

    async def move(self, component_name, destination, **kw):
        self.moves.append((component_name, destination, kw.get("constraints")))
        return True

    async def get_pose(self, component_name, destination_frame, **kw):
        return PoseInFrame(
            reference_frame=destination_frame,
            pose=Pose(x=self.tool_x, y=0, z=0, o_x=0, o_y=0, o_z=-1, theta=0),
        )

    @property
    def moved(self):
        return None if not self.moves else self.moves[-1][:2]


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


def _blank_images():
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    depth = np.full((480, 640), 700, dtype=np.uint16)
    return [
        _NamedImage(CameraMimeType.JPEG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ]


def _make_control(attrs=None, motion=None):
    ctrl = Control.__new__(Control)
    base = {"camera": "cam", "arm": "arm", "tool_frame": "tool", "blade_frame": "blade"}
    base.update(attrs or {})
    ctrl.settings = Settings.from_config(_config(base))
    ctrl.camera = _FakeCamera(_images_with_box())
    ctrl.motion = motion or _RecordingMotion()
    ctrl.arm = None
    ctrl.robot_client = _FakeRobotClient()
    ctrl._box_override = None
    ctrl._box_data = None
    return ctrl


# --- find_center --------------------------------------------------------------

@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_find_center_returns_world_pose():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "find_center"})
    assert out["found"] is True
    assert abs(out["u"] - 320) <= 5
    assert abs(out["v"] - 240) <= 5
    assert out["depth_mm"] == pytest.approx(700.0)
    assert out["override_applied"] is False
    assert abs(out["world_pose"]["x"]) < 5
    assert abs(out["world_pose"]["y"]) < 5
    assert out["world_pose"]["z"] == pytest.approx(700.0, abs=1)
    assert "cut_endpoints_world" in out


@pytest.mark.asyncio
async def test_find_center_requires_set_box():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="set_box"):
        await ctrl.do_command({"command": "find_center"})


@pytest.mark.asyncio
async def test_find_center_derives_box_frame_from_set_box():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", **BOX_1})
    out = await ctrl.do_command({"command": "find_center"})
    s = ctrl.settings
    box = out["box_frame"]
    assert (out["u"], out["v"]) == (BOX_1["u"], BOX_1["v"])
    assert out["depth_mm"] == pytest.approx(BOX_1["depth_mm"])
    # The fake world transform passes the camera-frame point straight through.
    assert out["world_pose"]["x"] == pytest.approx(out["camera_frame_xyz"]["x"])
    assert out["world_pose"]["z"] == pytest.approx(BOX_1["depth_mm"])
    # The tool-frame transform adds a known offset to the camera-frame z.
    knife_to_top = BOX_1["depth_mm"] + _FakeRobotClient.TOOL_OFFSET[2]
    assert box["knife_tip_to_top_mm"] == pytest.approx(knife_to_top, abs=1)
    assert box["center_z_mm"] == pytest.approx(
        s.knife_tip_to_table_mm - knife_to_top - s.base_plate_height_mm, abs=1
    )
    assert box["height_mm"] == pytest.approx(BOX_1["box_height_mm"])
    assert box["flap_width_mm"] == pytest.approx(BOX_1["flap_width_mm"])


@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_find_center_reports_not_found_on_blank_frame():
    ctrl = _make_control()
    ctrl.camera = _FakeCamera(_blank_images())
    out = await ctrl.do_command({"command": "find_center"})
    assert out["found"] is False
    assert "reason" in out


@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_find_center_raises_on_depth_color_mismatch():
    ctrl = _make_control()
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr)
    depth = np.full((240, 320), 700, dtype=np.uint16)  # mismatched resolution
    ctrl.camera = _FakeCamera([
        _NamedImage(CameraMimeType.JPEG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ])
    with pytest.raises(ValueError, match="not aligned"):
        await ctrl.do_command({"command": "find_center"})


class _NoIntr:
    focal_x_px = 0.0
    focal_y_px = 0.0
    center_x_px = 320.0
    center_y_px = 240.0


class _NoIntrProps:
    intrinsic_parameters = _NoIntr()


class _NoIntrCamera(_FakeCamera):
    async def get_properties(self):
        return _NoIntrProps()


@pytest.mark.asyncio
async def test_find_center_raises_without_intrinsics():
    ctrl = _make_control()
    ctrl.camera = _NoIntrCamera(_images_with_box())
    with pytest.raises(ValueError, match="intrinsic"):
        await ctrl.do_command({"command": "find_center"})


# --- dispatch -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_do_command_unknown_raises():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="unknown command"):
        await ctrl.do_command({"command": "bogus"})


@pytest.mark.asyncio
async def test_do_command_missing_command_raises():
    ctrl = _make_control()
    with pytest.raises(ValueError):
        await ctrl.do_command({})


@pytest.mark.asyncio
async def test_do_command_rejects_unknown_seam():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="unknown seam"):
        await ctrl.do_command({"command": "cut", "seam": "sideways"})


# --- set_box ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_box_stores_explicit_measurements():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "set_box", **BOX_1})
    assert out["box"] == {
        "depth_mm": 535.0, "u": 360, "v": 223, "flap_width_mm": 120.0,
        "box_height_mm": 280.0,
    }


@pytest.mark.parametrize("preset", [
    "box_1",
    *(
        pytest.param(p, marks=pytest.mark.xfail(
            strict=True, raises=ValueError,
            reason="box_height_mm not yet measured for this preset (-1)",
        ))
        for p in sorted(BOX_PRESETS) if p != "box_1"
    ),
])
@pytest.mark.asyncio
async def test_set_box_accepts_a_preset(preset):
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "set_box", "preset": preset})
    depth_mm, _, _, flap_width_mm, box_height_mm = BOX_PRESETS[preset]
    assert out["box"]["depth_mm"] == pytest.approx(depth_mm)
    assert out["box"]["flap_width_mm"] == pytest.approx(flap_width_mm)
    assert out["box"]["box_height_mm"] == pytest.approx(box_height_mm)


@pytest.mark.asyncio
async def test_set_box_rejects_unknown_preset():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="unknown box preset"):
        await ctrl.do_command({"command": "set_box", "preset": "box_9"})


@pytest.mark.asyncio
async def test_set_box_rejects_partial_measurements():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="flap_width_mm"):
        await ctrl.do_command({"command": "set_box", "depth_mm": 500, "u": 1, "v": 2})


@pytest.mark.asyncio
async def test_set_box_rejects_nonpositive_values():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="must be positive"):
        await ctrl.do_command({
            "command": "set_box", **BOX_1, "depth_mm": 0,
        })


@pytest.mark.asyncio
async def test_set_box_clear_drops_the_override():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    cleared = await ctrl.do_command({"command": "set_box", "clear": True})
    assert cleared["cleared"] is True
    with pytest.raises(ValueError, match="set_box"):
        await ctrl.do_command({"command": "find_center"})


@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_set_box_clear_restores_detection():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    cleared = await ctrl.do_command({"command": "set_box", "clear": True})
    assert cleared["cleared"] is True
    out = await ctrl.do_command({"command": "find_center"})
    assert out["override_applied"] is False
    assert abs(out["u"] - 320) <= 5


@pytest.mark.asyncio
async def test_set_box_invalidates_the_stored_box_frame():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    await ctrl.do_command({"command": "find_center"})
    assert ctrl._box_data is not None
    await ctrl.do_command({"command": "set_box", **BOX_1})
    assert ctrl._box_data is None


# --- home / move_to_center ----------------------------------------------------

@pytest.mark.asyncio
async def test_home_moves_tool_to_configured_pose():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "home"})
    assert out["homed"] is True
    comp, dest, _ = ctrl.motion.moves[0]
    assert comp == "tool"
    assert dest.reference_frame == "world"
    assert (dest.pose.x, dest.pose.y, dest.pose.z) == pytest.approx(
        ctrl.settings.home_xyz
    )
    assert dest.pose.o_z == -1


@pytest.mark.asyncio
async def test_move_to_center_descends_in_the_tool_frame():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    out = await ctrl.do_command({"command": "move_to_center"})
    assert out["moved"] is True
    comp, dest, constraints = ctrl.motion.moves[-1]
    assert comp == "tool"
    assert dest.reference_frame == "tool"
    # Stops short of the box top by center_standoff_mm.
    expected_z = out["box_frame"]["knife_tip_to_top_mm"] - ctrl.settings.center_standoff_mm
    assert dest.pose.z == pytest.approx(expected_z)
    assert constraints is not None


@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_move_to_center_skips_move_when_not_found():
    ctrl = _make_control()
    ctrl.camera = _FakeCamera(_blank_images())
    out = await ctrl.do_command({"command": "move_to_center"})
    assert out["found"] is False
    assert ctrl.motion.moves == []


# --- converge -----------------------------------------------------------------

def _frame_with_seam_at(x):
    """Camera frame whose only vertical line sits at pixel column x."""
    bgr = np.full((480, 640, 3), 200, dtype=np.uint8)
    bgr[100:400, x - 1:x + 2] = 20
    ok, buf = cv2.imencode(".png", bgr)
    depth = np.full((480, 640), 700, dtype=np.uint16)
    return [
        _NamedImage(CameraMimeType.PNG, data=buf.tobytes()),
        _NamedImage(CameraMimeType.VIAM_RAW_DEPTH, depth=depth),
    ]


class _ServoCamera:
    """Box frames first, then seam frames that walk toward the blade column."""
    def __init__(self, seam_columns):
        self._seam_columns = list(seam_columns)
        self._index = 0

    async def get_images(self):
        if self._index >= len(self._seam_columns):
            column = self._seam_columns[-1]
        else:
            column = self._seam_columns[self._index]
        self._index += 1
        return _frame_with_seam_at(column), None

    async def get_properties(self):
        return _Props()


async def _control_with_box_frame(motion=None, attrs=None):
    ctrl = _make_control(attrs=attrs, motion=motion)
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    await ctrl.do_command({"command": "find_center"})
    return ctrl


@pytest.mark.asyncio
async def test_converge_requires_a_box_frame():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "converge"})
    assert out["success"] is False
    assert "find_center" in out["reason"]


@pytest.mark.asyncio
async def test_converge_succeeds_once_the_seam_is_under_the_blade():
    ctrl = await _control_with_box_frame()
    b = int(ctrl.settings.blade_x_px)
    ctrl.camera = _ServoCamera([b + 21, b + 11, b])
    out = await ctrl.do_command({"command": "converge"})
    assert out["success"] is True
    assert out["seam"] == SEAM_TOP
    assert out["iterations"] == 3
    assert abs(out["error_px"]) <= ctrl.settings.converge_tolerance_px


@pytest.mark.asyncio
async def test_converge_converges_immediately_without_moving():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    ctrl.camera = _ServoCamera([int(ctrl.settings.blade_x_px)])
    out = await ctrl.do_command({"command": "converge"})
    assert out["success"] is True
    assert out["iterations"] == 1
    assert ctrl.motion.moves == []


@pytest.mark.asyncio
async def test_converge_steps_the_tool_toward_the_seam():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    b = int(ctrl.settings.blade_x_px)
    ctrl.camera = _ServoCamera([b + 21, b])
    out = await ctrl.do_command({"command": "converge"})
    assert out["success"] is True
    comp, dest, _ = ctrl.motion.moves[0]
    assert comp == "tool"
    assert dest.reference_frame == "tool"
    # Seam right of the blade (+21px). With the default Jacobian only tool y
    # moves the seam (+3.2 px/mm), so gain 0.2 gives a small negative y step.
    assert dest.pose.x == pytest.approx(0.0)
    assert -10 < dest.pose.y < 0


@pytest.mark.asyncio
async def test_converge_gives_up_after_max_iterations():
    ctrl = await _control_with_box_frame(attrs={"converge_max_iterations": 4})
    ctrl.camera = _ServoCamera([365])  # never moves under the blade
    out = await ctrl.do_command({"command": "converge"})
    assert out["success"] is False
    assert out["iterations"] == 4
    assert "converge_max_iterations" in out["reason"]


@pytest.mark.asyncio
async def test_converge_gives_up_when_no_seam_is_visible():
    ctrl = await _control_with_box_frame(attrs={"converge_max_blank_frames": 3})
    ctrl.camera = _FakeCamera(_blank_images())
    out = await ctrl.do_command({"command": "converge"})
    assert out["success"] is False
    assert out["iterations"] == 3
    assert "no seam line visible" in out["reason"]


@pytest.mark.asyncio
async def test_move_to_seam_stages_the_far_side_seam():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "move_to_seam", "seam": SEAM_FAR})
    assert out["moved"] is True

    stage, blade, offset = ctrl.motion.moves
    s, box = ctrl.settings, ctrl._box_data
    assert stage[1].reference_frame == "world"
    assert stage[1].pose.x == pytest.approx(s.stopper_x_mm + box.height_mm)
    assert stage[1].pose.y == pytest.approx(box.center_y_mm)
    assert stage[1].pose.z == pytest.approx(box.center_z_mm + s.side_seam_z_offset_mm)
    assert stage[1].pose.theta == pytest.approx(SIDE_SEAM_THETA_DEG)
    assert blade[0] == "blade"
    assert blade[1].pose.theta == pytest.approx(-s.blade_angle_deg)
    assert offset[1].pose.x == pytest.approx(
        -CUT_SIGN * s.seam_offset_fraction * box.flap_width_mm
    )
    assert offset[1].pose.y == pytest.approx(FAR_SEAM_APPROACH_LATERAL_MM)


@pytest.mark.asyncio
async def test_move_to_seam_stages_the_close_side_seam_at_the_stopper():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "move_to_seam", "seam": SEAM_CLOSE})
    assert out["moved"] is True
    stage, blade, offset = ctrl.motion.moves
    assert stage[1].pose.x == pytest.approx(ctrl.settings.stopper_x_mm)
    assert blade[1].pose.theta == pytest.approx(ctrl.settings.blade_angle_deg)
    assert offset[1].pose.y == pytest.approx(0.0)


@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_converge_refuses_a_side_seam_without_flap_width():
    ctrl = _make_control()
    ctrl.camera = _FakeCamera(_images_with_box())
    await ctrl.do_command({"command": "find_center"})  # no set_box -> no flap width
    out = await ctrl.do_command({"command": "converge", "seam": SEAM_FAR})
    assert out["success"] is False
    assert "flap width" in out["reason"]


# --- cut ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cut_requires_a_box_frame():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_TOP})
    assert out["completed"] is False
    assert "find_center" in out["reason"]


@pytest.mark.asyncio
async def test_cut_top_slices_both_halves_from_the_center():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_TOP})
    assert out["completed"] is True
    assert out["steps"] == [
        "insert", "slice_forward", "slice_forward", "slice_forward", "retract",
        "return_to_center",
        "insert", "slice_back", "slice_back", "slice_back", "retract",
    ]

    s, box = ctrl.settings, ctrl._box_data
    poses = [dest.pose for _, dest, _ in ctrl.motion.moves]
    span = CUT_SIGN * s.top_seam_span_fraction * box.height_mm
    assert poses[0].z == pytest.approx(s.top_blade_insert_mm)
    # The three forward chunks sum to the configured span (past the box edge,
    # on purpose, to clear the flaps)...
    assert sum(p.x for p in poses[1:4]) == pytest.approx(span)
    # ...and the return move undoes exactly that.
    assert poses[5].x == pytest.approx(-span)
    assert sum(p.x for p in poses[7:10]) == pytest.approx(-span)
    assert poses[-1].z == pytest.approx(-s.top_blade_insert_mm)
    # The top seam is cut freehand; only the side seams are line-constrained.
    assert all(c is None for _, _, c in ctrl.motion.moves)


@pytest.mark.asyncio
async def test_cut_far_seam_slices_under_a_linear_constraint():
    # side_blade_insert_mm defaults to 0 while it is being tuned; set it so the
    # insert and retract are distinguishable.
    ctrl = await _control_with_box_frame(attrs={"side_blade_insert_mm": 16})
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_FAR})
    assert out["completed"] is True
    assert out["steps"] == ["insert", "slice", "retract", "straighten_blade"]

    s = ctrl.settings
    insert, slice_move, retract, straighten = ctrl.motion.moves
    assert insert[1].pose.z == pytest.approx(s.side_blade_insert_mm)
    assert slice_move[1].pose.x == pytest.approx(CUT_SIGN * s.side_seam_slice_mm)
    assert slice_move[2] is not None  # linear constraint on the cut itself
    assert retract[1].pose.z == pytest.approx(-s.side_blade_insert_mm)
    assert straighten[0] == "blade"
    assert straighten[1].pose.theta == pytest.approx(s.blade_angle_deg)


@pytest.mark.asyncio
async def test_cut_close_seam_retracts_clear():
    ctrl = await _control_with_box_frame(attrs={"side_blade_insert_mm": 16})
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_CLOSE})
    assert out["steps"] == [
        "insert", "slice", "retract", "straighten_blade", "straighten_tool",
    ]
    s = ctrl.settings
    moves = ctrl.motion.moves
    assert moves[0][1].pose.z == pytest.approx(s.side_blade_insert_mm)
    assert moves[2][1].pose.z == pytest.approx(-40.0)
    assert moves[3][1].pose.theta == pytest.approx(-s.blade_angle_deg)
    assert moves[4][1].pose.theta == pytest.approx(90.0)


# --- seam resolution from pose -------------------------------------------------

@pytest.mark.asyncio
async def test_cut_infers_the_top_seam_from_the_tool_pose():
    motion = _RecordingMotion()
    ctrl = await _control_with_box_frame(motion=motion)
    motion.tool_x = ctrl._box_data.center_x_mm
    out = await ctrl.do_command({"command": "cut"})
    assert out["seam"] == SEAM_TOP


@pytest.mark.asyncio
async def test_cut_infers_the_far_seam_from_the_tool_pose():
    motion = _RecordingMotion()
    ctrl = await _control_with_box_frame(motion=motion)
    motion.tool_x = ctrl.settings.stopper_x_mm + ctrl._box_data.height_mm
    out = await ctrl.do_command({"command": "cut"})
    assert out["seam"] == SEAM_FAR


@pytest.mark.asyncio
async def test_cut_infers_the_close_seam_from_the_tool_pose():
    motion = _RecordingMotion()
    ctrl = await _control_with_box_frame(motion=motion)
    motion.tool_x = ctrl.settings.stopper_x_mm + 3
    out = await ctrl.do_command({"command": "cut"})
    assert out["seam"] == SEAM_CLOSE


@pytest.mark.asyncio
async def test_cut_refuses_when_the_tool_is_at_no_seam():
    motion = _RecordingMotion(tool_x=5000.0)
    ctrl = await _control_with_box_frame(motion=motion)
    motion.moves.clear()
    out = await ctrl.do_command({"command": "cut"})
    assert out["completed"] is False
    assert "nearest seam" in out["reason"]
    assert motion.moves == []


@pytest.mark.asyncio
async def test_cut_refuses_when_two_seams_are_equally_close():
    motion = _RecordingMotion()
    ctrl = await _control_with_box_frame(
        motion=motion, attrs={"seam_match_tolerance_mm": 300}
    )
    s, box = ctrl.settings, ctrl._box_data
    # Halfway between the close seam and the top seam.
    motion.tool_x = (s.stopper_x_mm + box.center_x_mm) / 2
    motion.moves.clear()
    out = await ctrl.do_command({"command": "cut"})
    assert out["completed"] is False
    assert "cannot tell" in out["reason"]
    assert motion.moves == []


# --- full_cut -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_cut_runs_home_center_and_all_three_seams():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    ctrl.camera = _ServoCamera([int(ctrl.settings.blade_x_px)])
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["completed"] is True
    assert out["steps"] == [
        "home", "move_to_center",
        "converge_top", "cut_top",
        "converge_far", "cut_far",
        "converge_close", "cut_close",
        "home",
    ]
    assert [s["seam"] for s in out["seams"]] == [SEAM_TOP, SEAM_FAR, SEAM_CLOSE]


@DETECTION_DISABLED
@pytest.mark.asyncio
async def test_full_cut_aborts_when_no_box():
    ctrl = _make_control()
    ctrl.camera = _FakeCamera(_blank_images())
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["completed"] is False
    assert out["found"] is False
    assert out["steps"] == ["home"]


@pytest.mark.asyncio
async def test_full_cut_stops_at_the_seam_that_fails_to_converge():
    ctrl = _make_control(attrs={"converge_max_blank_frames": 2})
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    # Detection succeeds on the box frames, but no seam line is ever visible.
    ctrl.camera = _FakeCamera(_images_with_box())
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["completed"] is False
    assert out["failed_at"] == SEAM_TOP
    assert out["stage"] == "converge"
    assert out["steps"] == ["home", "move_to_center"]


# --- stop ---------------------------------------------------------------------

class _BlockingMotion(_RecordingMotion):
    """A move that never finishes on its own, like a long cut stroke."""
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def move(self, component_name, destination, **kw):
        self.moves.append((component_name, destination, kw.get("constraints")))
        self.started.set()
        await asyncio.Event().wait()


class _FakeArm:
    def __init__(self):
        self.stops = 0

    async def stop(self, **kw):
        self.stops += 1


@pytest.mark.asyncio
async def test_stop_interrupts_a_running_command_and_halts_the_arm():
    motion = _BlockingMotion()
    ctrl = _make_control(motion=motion)
    ctrl.arm = _FakeArm()
    running = asyncio.create_task(ctrl.do_command({"command": "home"}))
    await motion.started.wait()

    with pytest.raises(ValueError, match="busy running 'home'"):
        await ctrl.do_command({"command": "jog", "x": 1})

    out = await ctrl.do_command({"command": "stop"})
    assert out == {"stopped": True, "interrupted": "home"}
    assert ctrl.arm.stops == 1
    assert await running == {"stopped": True, "command": "home"}
    assert len(motion.moves) == 1  # nothing moved after the stop
    assert ctrl._task is None


@pytest.mark.asyncio
async def test_stop_with_nothing_running_still_halts_the_arm():
    ctrl = _make_control()
    ctrl.arm = _FakeArm()
    out = await ctrl.do_command({"command": "stop"})
    assert out == {"stopped": True, "interrupted": None}
    assert ctrl.arm.stops == 1


# --- dry run ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_cut_without_dry_run_adds_no_transforms():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    ctrl.camera = _ServoCamera([int(ctrl.settings.blade_x_px)])
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["completed"] is True
    # Only find_center's two transforms (camera -> world, camera -> tool).
    assert len(ctrl.robot_client.requests) == 2
    assert "dry_run" not in out
    # every move, blade inserts and retracts included
    assert len(ctrl.motion.moves) == 29


@pytest.mark.asyncio
async def test_dry_run_top_cut_skips_the_blade_in_and_out():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_TOP, "dry_run": True})
    assert out["completed"] is True
    assert out["dry_run"] is True
    assert out["skipped"] == ["top:insert", "top:retract", "top:insert", "top:retract"]
    # 3 forward chunks, the return to center, 3 back chunks -- all level.
    assert len(ctrl.motion.moves) == 7
    assert all(dest.pose.z == 0 for _, dest, _ in ctrl.motion.moves)


@pytest.mark.asyncio
async def test_dry_run_close_retract_keeps_the_extra_pull_back():
    ctrl = await _control_with_box_frame(attrs={"side_blade_insert_mm": 16})
    ctrl.motion.moves.clear()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_CLOSE, "dry_run": True})
    assert out["skipped"] == ["close:insert"]
    slice_move, retract, straighten_blade, straighten_tool = ctrl.motion.moves
    # The insert was skipped, so only the clearance beyond it is pulled back.
    assert retract[1].pose.z == pytest.approx(-(CLOSE_SEAM_RETRACT_MM - 16))


@pytest.mark.asyncio
async def test_dry_run_full_cut_lists_every_skipped_move():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    ctrl.camera = _ServoCamera([int(ctrl.settings.blade_x_px)])
    out = await ctrl.do_command({"command": "full_cut", "dry_run": True})
    assert out["completed"] is True
    assert out["skipped"] == [
        "top:insert", "top:retract", "top:insert", "top:retract",
        "far:insert", "far:retract",
        "close:insert",
    ]
    assert out["bounds_checked"] is False  # no workspace configured


@pytest.mark.asyncio
async def test_dry_run_is_ignored_on_commands_that_do_not_move():
    ctrl = _make_control()
    out = await ctrl.do_command({"command": "set_box", "preset": "box_1", "dry_run": True})
    assert "dry_run" not in out
    assert ctrl._dry_run is False


@pytest.mark.asyncio
async def test_dry_run_must_be_a_bool():
    ctrl = _make_control()
    with pytest.raises(ValueError, match="'dry_run' must be true or false"):
        await ctrl.do_command({"command": "home", "dry_run": "false"})
    assert ctrl.motion.moves == []


@pytest.mark.asyncio
async def test_stopped_dry_run_does_not_leak_into_the_next_real_run():
    motion = _BlockingMotion()
    ctrl = await _control_with_box_frame(motion=motion)
    ctrl.arm = _FakeArm()
    running = asyncio.create_task(
        ctrl.do_command({"command": "cut", "seam": SEAM_TOP, "dry_run": True})
    )
    await motion.started.wait()
    await ctrl.do_command({"command": "stop"})
    assert (await running)["stopped"] is True
    assert ctrl._dry_run is False

    ctrl.motion = _RecordingMotion()
    out = await ctrl.do_command({"command": "cut", "seam": SEAM_TOP})
    assert "dry_run" not in out
    # The real cut sends its inserts and retracts: 4 z moves plus 7 slices.
    assert len(ctrl.motion.moves) == 11


@pytest.mark.asyncio
async def test_dry_run_move_to_center_stops_higher():
    ctrl = _make_control()
    await ctrl.do_command({"command": "set_box", "preset": "box_1"})
    out = await ctrl.do_command({"command": "move_to_center", "dry_run": True})
    s = ctrl.settings
    _, dest, _ = ctrl.motion.moves[-1]
    assert dest.pose.z == pytest.approx(
        out["box_frame"]["knife_tip_to_top_mm"]
        - s.center_standoff_mm
        - s.dry_run_clearance_mm
    )


@pytest.mark.asyncio
async def test_dry_run_side_seam_approach_stops_higher():
    ctrl = await _control_with_box_frame()
    ctrl.motion.moves.clear()
    await ctrl.do_command({"command": "move_to_seam", "seam": SEAM_FAR, "dry_run": True})
    s, box = ctrl.settings, ctrl._box_data
    stage = ctrl.motion.moves[0]
    assert stage[1].pose.z == pytest.approx(
        box.center_z_mm + s.side_seam_z_offset_mm + s.dry_run_clearance_mm
    )
