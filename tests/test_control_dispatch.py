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
    ctrl = Control.__new__(Control)
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
    with pytest.raises(ValueError):
        await ctrl.do_command({})


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
    assert comp == "tool"
    assert dest.reference_frame == "world"
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
            return self._home
        return await super().get_pose(component_name, destination_frame, supplemental_transforms)


@pytest.mark.asyncio
async def test_full_cut_runs_full_sequence():
    ctrl = _make_control()
    ctrl.motion = _FullCutMotion()
    ctrl.arm = _FakeArm()
    out = await ctrl.do_command({"command": "full_cut"})
    assert out["completed"] is True
    assert ctrl.arm.joint_history[0][ctrl.settings.twist_joint_index] == ctrl.settings.twist_angle_deg
    assert ctrl.arm.joint_history[-1][ctrl.settings.twist_joint_index] == 0
    assert len(ctrl.motion.moves) >= 5
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
