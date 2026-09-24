"""The `viam-labs:box-cutter:control` service.

The control service exposes a number of DoCommand endpoints that allow the
user to operate an arm equipped with a camera in a box-cutting scenario.

Seam names throughout are from the blade's point of view at the box:
  top   -- the seam running across the top of the box, cut in two passes
  far   -- the side seam on the far edge, away from the stopper
  close -- the side seam at the stopper, nearest the robot base
"""

import asyncio
import os
from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.proto.service.motion import LinearConstraint
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.robot.client import RobotClient
from viam.services.generic import Generic
from viam.services.motion import Constraints, MotionClient
from viam.utils import ValueTypes

from models.detection import (
    decode_color,
    deproject,
    detect_box_center,
    find_seam_edges,
    find_vertical_seam_line,
    inset_endpoints,
    invert_jacobian,
    pixel_error_to_delta_mm,
    sample_depth_in_mask,
)

SEAM_TOP = "top"
SEAM_FAR = "far"
SEAM_CLOSE = "close"
SEAMS = (SEAM_TOP, SEAM_FAR, SEAM_CLOSE)

# Hand-tuned asymmetries from expirmental data that adjust the motion calls of the arm.
CLOSE_SEAM_RETRACT_MM = 40.0      # close seam retracts much further than it inserted
FAR_SEAM_APPROACH_LATERAL_MM = 5.0  # lateral nudge only the far approach uses
# Which way along tool x the blade travels when slicing. The tool frame's x runs
# along the seam at every seam (the tool yaws with the seam), but whether +x
# points up or down the seam was not derivable from the frame measurements --
# flip this if a retracted dry run shows the side-seam stroke leaving the box.
# The top seam is unaffected: it cuts symmetrically either side of center.
CUT_SIGN = 1.0
CLOSE_SEAM_FINAL_THETA_DEG = 90.0  # unwinds the tool after the last cut
# TODO: still the old arm's value. This is a world-frame yaw (passed to
# `_world_pose`), so the +90 deg base rotation shifts it to either 0 or -180.
# Theta is measured about o_z=-1, which makes the sign easy to get backwards --
# jog the tool into the side-seam orientation and read it back from
# `motion.get_pose(tool_frame, world)` rather than deriving it.
SIDE_SEAM_THETA_DEG = -90.0       # tool yaw for both side-seam approaches

# Two seam candidates closer together than this cannot be told apart from the
# tool pose alone, so `cut` refuses rather than guessing which one it is on.
SEAM_AMBIGUITY_MM = 5.0

# `jog` is driven by hand with a blade mounted, so a fat-fingered 50 where 5 was
# meant should be refused rather than executed.
JOG_MAX_MM = 50.0


# TODO: remeasure the cut calculations (height_mm for the box)
# The five boxes measured on the original cell, as
# (depth_mm, u, v, flap_width_mm, box_height_mm) -- see `set_box`.
BOX_PRESETS = {
    "box_1": (535.0, 360, 223, 120.0, 280.0),
    "box_2": (456.0, 401, 254, 106.0, -1),
    "box_3": (479.0, 422, 262, 80.0, -1),
    "box_4": (477.0, 425, 285, 82.0, -1),
    "box_5": (553.0, 413, 260, 80.0, -1),
}

_BOX_MEASUREMENTS = ("depth_mm", "u", "v", "flap_width_mm", "box_height_mm")


async def create_robot_client_from_module():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_FQDN"], opts)


def _str(
    config: ComponentConfig, key: str, default: Optional[str] = None
) -> Optional[str]:
    fields = config.attributes.fields
    if key in fields and fields[key].string_value:
        return fields[key].string_value
    return default


def _num(config: ComponentConfig, key: str, default):
    fields = config.attributes.fields
    if key in fields and fields[key].HasField("number_value"):
        return type(default)(fields[key].number_value)
    return default


def _triple(
    config: ComponentConfig, key: str, default: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    fields = config.attributes.fields
    if key in fields and fields[key].HasField("list_value"):
        vals = [int(v.number_value) for v in fields[key].list_value.values]
        if len(vals) != 3:
            raise ValueError(f"'{key}' must have exactly 3 numbers")
        return (vals[0], vals[1], vals[2])
    return default


def _floats(
    config: ComponentConfig,
    key: str,
    default: Optional[Tuple[float, ...]],
    length: Optional[int] = None,
) -> Optional[Tuple[float, ...]]:
    """A configured list of floats, optionally of a fixed length."""
    fields = config.attributes.fields
    if key not in fields or not fields[key].HasField("list_value"):
        return default
    vals = tuple(float(v.number_value) for v in fields[key].list_value.values)
    if length is not None and len(vals) != length:
        raise ValueError(f"'{key}' must have exactly {length} numbers")
    if not vals:
        raise ValueError(f"'{key}' must not be empty")
    return vals


def _properties_to_dict(properties) -> dict:
    """Camera properties flattened to JSON-safe primitives.

    Read defensively: a DoCommand result has to serialize into a protobuf
    Struct, and which fields a camera populates varies by model -- a webcam with
    no calibration omits the intrinsics a RealSense reports.
    """
    out = {
        "supports_pcd": bool(getattr(properties, "supports_pcd", False)),
        "mime_types": [str(m) for m in getattr(properties, "mime_types", [])],
    }

    frame_rate = getattr(properties, "frame_rate", None)
    if frame_rate is not None:
        out["frame_rate"] = float(frame_rate)

    intr = getattr(properties, "intrinsic_parameters", None)
    if intr is not None:
        out["intrinsic_parameters"] = {
            "width_px": int(getattr(intr, "width_px", 0)),
            "height_px": int(getattr(intr, "height_px", 0)),
            "fx": float(getattr(intr, "focal_x_px", 0.0)),
            "fy": float(getattr(intr, "focal_y_px", 0.0)),
            "ppx": float(getattr(intr, "center_x_px", 0.0)),
            "ppy": float(getattr(intr, "center_y_px", 0.0)),
        }

    dist = getattr(properties, "distortion_parameters", None)
    if dist is not None:
        out["distortion_parameters"] = {
            "model": str(getattr(dist, "model", "")),
            "parameters": [float(p) for p in getattr(dist, "parameters", [])],
        }
    return out


def _pose_to_dict(pose) -> dict:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "o_x": float(pose.o_x),
        "o_y": float(pose.o_y),
        "o_z": float(pose.o_z),
        "theta": float(pose.theta),
    }


@dataclass
class BoxData:
    """Everything the cutting commands need about the box in front of the blade.

    Derived once by `find_center` and read by every later command, so a converge
    or a cut never re-detects mid-sequence (and never disagrees with the pass
    that positioned the tool).
    """

    center_x_mm: float          # box center, world frame
    center_y_mm: float
    center_z_mm: float
    height_mm: float            # how tall the box reads in the camera FOV
    knife_tip_to_top_mm: float  # tool-frame z of the box top
    tool_x_mm: float            # box center expressed in the tool frame
    tool_y_mm: float
    flap_width_mm: Optional[float]  # center-to-side; only an operator knows it

    def to_dict(self) -> dict:
        return {
            "center_x_mm": self.center_x_mm,
            "center_y_mm": self.center_y_mm,
            "center_z_mm": self.center_z_mm,
            "height_mm": self.height_mm,
            "knife_tip_to_top_mm": self.knife_tip_to_top_mm,
            "flap_width_mm": self.flap_width_mm,
        }


@dataclass
class Settings:
    camera_name: str
    arm_name: str
    tool_frame: str
    blade_frame: str
    motion_name: str
    camera_frame: str
    world_frame: str

    # Box detection.
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    min_box_area: int
    inset_mm: float
    min_seam_len_px: int
    seam_dark_v_max: int

    # Measured ground truth for this cell.
    stopper_x_mm: float
    knife_tip_to_table_mm: float
    base_plate_height_mm: float
    home_xyz: Tuple[float, ...]
    center_standoff_mm: float

    # Visual servoing.
    blade_x_px: float
    converge_tolerance_px: float
    servo_jacobian: Tuple[float, ...]
    top_seam_gain: float
    far_seam_gain: float
    close_seam_gain: float
    converge_max_iterations: int
    converge_max_blank_frames: int
    seam_search_radius_px: float

    # Cutting geometry.
    top_blade_insert_mm: float
    side_blade_insert_mm: float
    top_seam_chunks: Tuple[float, ...]
    side_seam_slice_mm: float
    side_seam_z_offset_mm: float
    blade_angle_deg: float
    seam_offset_fraction: float
    seam_match_tolerance_mm: float
    # Linear tolerances for the two constrained moves: the descent onto the box
    # top wants to stay roughly vertical, the cut itself wants to stay on the seam.
    descent_tolerance_mm: float
    cut_tolerance_mm: float

    # Dry run: only read when a command passes `dry_run`.
    dry_run_clearance_mm: float
    workspace_min_xyz: Optional[Tuple[float, ...]]
    workspace_max_xyz: Optional[Tuple[float, ...]]

    @property
    def top_seam_span_fraction(self) -> float:
        """Fraction of the box height one top-seam pass covers."""
        return sum(self.top_seam_chunks)

    def jacobian_rows(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        a, b, c, d = self.servo_jacobian
        return ((a, b), (c, d))

    def gain_for(self, seam: str) -> float:
        """Servo gain for one seam. The script tuned each approach separately."""
        if seam == SEAM_TOP:
            return self.top_seam_gain
        if seam == SEAM_FAR:
            return self.far_seam_gain
        return self.close_seam_gain

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

        # A negative clearance would make a dry run pass closer to the box than
        # a real run.
        dry_run_clearance_mm = _num(config, "dry_run_clearance_mm", 30.0)
        if dry_run_clearance_mm < 0:
            raise ValueError("'dry_run_clearance_mm' must not be negative")

        # The bounds are optional, but a half-set or inverted box is a config
        # mistake that would otherwise surface as every dry-run move failing.
        workspace_min_xyz = _floats(config, "workspace_min_xyz", None, length=3)
        workspace_max_xyz = _floats(config, "workspace_max_xyz", None, length=3)
        if (workspace_min_xyz is None) != (workspace_max_xyz is None):
            raise ValueError(
                "'workspace_min_xyz' and 'workspace_max_xyz' must be set together"
            )
        if workspace_min_xyz is not None:
            inverted = [
                axis
                for axis, lo, hi in zip("xyz", workspace_min_xyz, workspace_max_xyz)
                if lo > hi
            ]
            if inverted:
                raise ValueError(
                    "'workspace_min_xyz' exceeds 'workspace_max_xyz' on "
                    + ", ".join(inverted)
                )
        return cls(
            camera_name=camera_name,
            arm_name=arm_name,
            tool_frame=tool_frame,
            blade_frame=_str(config, "blade_frame", "blade-tool-90deg"),
            motion_name=_str(config, "motion_service", "builtin"),
            camera_frame=_str(config, "camera_frame", camera_name),
            world_frame=_str(config, "world_frame", "world"),
            hsv_lower=_triple(config, "hsv_lower", (10, 60, 80)),
            hsv_upper=_triple(config, "hsv_upper", (35, 255, 255)),
            min_box_area=_num(config, "min_box_area", 5000),
            inset_mm=_num(config, "inset_mm", 8.0),
            min_seam_len_px=_num(config, "min_seam_len_px", 60),
            seam_dark_v_max=_num(config, "seam_dark_v_max", 80),
            stopper_x_mm=_num(config, "stopper_x_mm", 370.0),
            knife_tip_to_table_mm=_num(config, "knife_tip_to_table_mm", 490.0),
            base_plate_height_mm=_num(config, "base_plate_height_mm", 20.0),
            home_xyz=_floats(config, "home_xyz", (399.97, -0, 406.48), length=3),
            center_standoff_mm=_num(config, "center_standoff_mm", 20.0),
            blade_x_px=_num(config, "blade_x_px", 344.0),
            converge_tolerance_px=_num(config, "converge_tolerance_px", 2.25),
            # Measured on this cell at the top-seam standoff: tool +y moves the
            # seam +3.2 px per mm. The dv/dX term only has to be nonzero for the
            # matrix to invert -- converge hardcodes error_v to 0, so it never
            # reaches the output.
            servo_jacobian=_floats(
                config, "servo_jacobian", (0.0, 3.2, -3.2, 0.0), length=4
            ),
            top_seam_gain=_num(config, "top_seam_gain", 0.2),
            far_seam_gain=_num(config, "far_seam_gain", 0.05),
            close_seam_gain=_num(config, "close_seam_gain", 0.09),
            converge_max_iterations=_num(config, "converge_max_iterations", 25),
            converge_max_blank_frames=_num(config, "converge_max_blank_frames", 5),
            seam_search_radius_px=_num(config, "seam_search_radius_px", 40.0),
            top_blade_insert_mm=_num(config, "top_blade_insert_mm", 25.0),
            side_blade_insert_mm=_num(config, "side_blade_insert_mm", 0.0), # was 16
            top_seam_chunks=_floats(config, "top_seam_chunks", (0.2, 0.2, 0.25)),
            side_seam_slice_mm=_num(config, "side_seam_slice_mm", 90.0),
            side_seam_z_offset_mm=_num(config, "side_seam_z_offset_mm", 15.0),
            blade_angle_deg=_num(config, "blade_angle_deg", 30.0),
            seam_offset_fraction=_num(config, "seam_offset_fraction", 0.45),
            seam_match_tolerance_mm=_num(config, "seam_match_tolerance_mm", 40.0),
            descent_tolerance_mm=_num(config, "descent_tolerance_mm", 10.0),
            cut_tolerance_mm=_num(config, "cut_tolerance_mm", 3.0),
            dry_run_clearance_mm=dry_run_clearance_mm,
            workspace_min_xyz=workspace_min_xyz,
            workspace_max_xyz=workspace_max_xyz,
        )


class Control(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "box-cutter"), "control")

    # Opened lazily by `_to_frame` and kept across reconfigures: it is built from
    # the module's environment, which a reconfigure does not change.
    robot_client: Optional[RobotClient] = None
    # The command currently driving the arm, so `stop` can cancel it.
    _task: Optional[asyncio.Task] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        cls = super().new(config, dependencies)
        cls.reconfigure(config, dependencies)
        return cls

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        settings = Settings.from_config(config)
        required = [settings.camera_name, settings.arm_name, settings.motion_name]
        return required, []

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
        self.camera: Camera = self._resolve(
            dependencies, Camera.get_resource_name(settings.camera_name)
        )
        self.arm: Arm = self._resolve(
            dependencies, Arm.get_resource_name(settings.arm_name)
        )
        self.motion: MotionClient = self._resolve(
            dependencies, MotionClient.get_resource_name(settings.motion_name)
        )
        # Per-box state cannot outlive a reconfigure: new geometry would be read
        # against measurements taken for the previous setup.
        self._box_override: Optional[dict] = None
        self._box_data: Optional[BoxData] = None

    # --- dispatch -------------------------------------------------------------

    async def get_properties(self) -> Mapping[str, ValueTypes]:
        """The camera's reported properties, for reading calibration off a cell.

        Note the API and the camera config spell intrinsics differently: the
        response gives focal_x_px/focal_y_px/center_x_px/center_y_px, while a
        camera config wants fx/fy/ppx/ppy. Unknown config keys are dropped
        silently, so a straight paste validates but yields no intrinsics.
        """
        properties = await self.camera.get_properties()
        return _properties_to_dict(properties)

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
        if name == "stop":
            return await self.stop()
        if name == "get_properties":
            return await self.get_properties()

        # One command drives the arm at a time: two interleaved sequences would
        # issue each other's moves, and `stop` could only cancel one of them.
        if self._task is not None:
            raise ValueError(
                f"busy running {self._task.get_name()!r}; send 'stop' first"
            )
        task = asyncio.create_task(self._dispatch(name, command), name=name)
        self._task = task
        try:
            # `wait` rather than `await task`, so a `stop` cancelling the task
            # comes back as a result instead of cancelling this caller too.
            await asyncio.wait([task])
        except asyncio.CancelledError:
            # The caller went away (e.g. the client disconnected): don't leave
            # the arm running a sequence nobody is watching.
            task.cancel()
            raise
        finally:
            if self._task is task:
                self._task = None
        if task.cancelled():
            return {"stopped": True, "command": name}
        return task.result()

    async def _dispatch(
        self, name: str, command: Mapping[str, ValueTypes]
    ) -> Mapping[str, ValueTypes]:
        seam = self._seam_arg(command)
        if name == "set_box":
            return self.set_box(command)
        if name == "home":
            return await self.home()
        if name == "jog":
            return await self.jog(command)
        if name == "find_center":
            return await self.find_center()
        if name == "move_to_center":
            return await self.move_to_center()
        if name == "converge":
            return await self.converge(seam or SEAM_TOP)
        if name == "cut":
            return await self.cut(seam)
        if name == "full_cut":
            return await self.full_cut()
        if name == "tool_change":
            return await self.tool_change()
        if name == "move_to_seam":
            return await self.move_to_seam(seam)
        raise ValueError(f"unknown command: {name!r}")

    @staticmethod
    def _seam_arg(command: Mapping[str, ValueTypes]) -> Optional[str]:
        seam = command.get("seam")
        if seam is None:
            return None
        seam = str(seam)
        if seam not in SEAMS:
            raise ValueError(
                f"unknown seam: {seam!r} (expected one of {', '.join(SEAMS)})"
            )
        return seam

    # --- set_box --------------------------------------------------------------

    def set_box(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        """Record the box measurements detection cannot recover on its own.

        Depth and center pixel are measured per box on this cell, and flap width
        (center to side) is not visible from above at all -- so the operator
        supplies them, either directly or by naming one of the presets.

        Any override invalidates the stored box frame: geometry derived from the
        previous numbers must not be read against new ones.
        """
        self._box_data = None

        if command.get("clear"):
            self._box_override = None
            return {"cleared": True}

        preset = command.get("preset")
        if preset is not None:
            key = str(preset)
            if key not in BOX_PRESETS:
                raise ValueError(
                    f"unknown box preset: {key!r} "
                    f"(known presets: {', '.join(sorted(BOX_PRESETS))})"
                )
            depth_mm, u, v, flap_width_mm, box_height_mm = BOX_PRESETS[key]
        else:
            missing = [k for k in _BOX_MEASUREMENTS if command.get(k) is None]
            if missing:
                raise ValueError(
                    "set_box needs a 'preset' or all of "
                    f"{', '.join(_BOX_MEASUREMENTS)}; missing: {', '.join(missing)}"
                )
            depth_mm = float(command["depth_mm"])
            u = int(command["u"])
            v = int(command["v"])
            flap_width_mm = float(command["flap_width_mm"])
            box_height_mm = float(command["box_height_mm"])

        box = {
            "depth_mm": float(depth_mm),
            "u": int(u),
            "v": int(v),
            "flap_width_mm": float(flap_width_mm),
            "box_height_mm": float(box_height_mm),
        }
        nonpositive = [k for k, val in box.items() if val <= 0]
        if nonpositive:
            raise ValueError(
                f"set_box measurements must be positive; got "
                f"{', '.join(f'{k}={box[k]}' for k in nonpositive)}"
            )

        self._box_override = box
        return {"box": box}

    # --- detection ------------------------------------------------------------

    # NOTE: potentially move overrides higher in this function
    async def find_center(self) -> Mapping[str, ValueTypes]:
        """Detect the box, apply any override, and derive the box frame."""
        s = self.settings
        # images, _ = await self.camera.get_images()
        properties = await self.camera.get_properties()
        intr = properties.intrinsic_parameters
        # A camera with no calibration reports zero focal lengths, which would
        # surface as a ZeroDivisionError from inside deproject rather than
        # something a caller can act on.
        if not intr.focal_x_px or not intr.focal_y_px:
            raise ValueError(
                "camera returned no intrinsic parameters; cannot deproject "
                "(configure the camera to emit intrinsics)"
            )

        # color_bgr, depth_np = decode_color_and_depth(images)
        # if depth_np.shape[:2] != color_bgr.shape[:2]:
        #     raise ValueError(
        #         f"depth {depth_np.shape[:2]} not aligned to color "
        #         f"{color_bgr.shape[:2]}; cannot deproject with color intrinsics"
        #     )

        # # A failed detection drops the stored frame: if the box cannot be seen,
        # # the geometry derived when it could must not be cut against.
        # u, v, mask = detect_box_center(
        #     color_bgr, s.hsv_lower, s.hsv_upper, s.min_box_area
        # )
        # if u is None:
        #     self._box_data = None
        #     return {"found": False, "reason": "no box-colored region found"}

        # z = sample_depth_in_mask(depth_np, mask)
        # if z is None:
        #     self._box_data = None
        #     return {"found": False, "reason": "no valid depth in box mask"}

        # # The seam overlay is reported from what was actually detected, before any
        # # override replaces the center -- it describes this frame, not the box.
        # seam = find_seam_edges(
        #     mask, (u, v), color_bgr, s.seam_dark_v_max, s.min_seam_len_px
        # )

        override = self._box_override
        if not override:
            raise ValueError("Currently only supporting manual setting. Run set_box first.")

        flap_width_mm = None
        height_mm: Optional[float] = None
        z = override["depth_mm"]
        u = override["u"]
        v = override["v"]
        flap_width_mm = override["flap_width_mm"]
        height_mm = override["box_height_mm"]

        cx, cy, cz = deproject(u, v, z, intr)
        world = await self._to_frame((cx, cy, cz), s.world_frame)
        tool = await self._to_frame((cx, cy, cz), s.tool_frame)

        if height_mm is None:
            height_mm = 2 * abs(s.stopper_x_mm - float(world.pose.x))

        # Ground truth: the stopper pins the near edge of the box, so the box
        # reaches as far past its own center again -- hence the doubling. The top
        # sits a knife-tip's reach below the table height, less the base plate.
        knife_tip_to_top_mm = float(tool.pose.z)
        box = BoxData(
            center_x_mm=float(world.pose.x),
            center_y_mm=float(world.pose.y),
            center_z_mm=(
                s.knife_tip_to_table_mm - knife_tip_to_top_mm - s.base_plate_height_mm
            ),
            height_mm=height_mm,
            knife_tip_to_top_mm=knife_tip_to_top_mm,
            tool_x_mm=float(tool.pose.x),
            tool_y_mm=float(tool.pose.y),
            flap_width_mm=flap_width_mm,
        )
        self._box_data = box

        result = {
            "found": True,
            "override_applied": override is not None,
            "u": int(u),
            "v": int(v),
            "depth_mm": float(z),
            "camera_frame_xyz": {"x": float(cx), "y": float(cy), "z": float(cz)},
            "world_pose": _pose_to_dict(world.pose),
            "tool_pose": _pose_to_dict(tool.pose),
            "box_frame": box.to_dict(),
        }

        # if seam is not None:
        #     top_px, bottom_px, angle_deg = seam
        #     top_w = await self._endpoint_world(top_px, z, intr)
        #     bottom_w = await self._endpoint_world(bottom_px, z, intr)
        #     top_inset, bottom_inset = inset_endpoints(top_w, bottom_w, s.inset_mm)
        #     result["seam"] = {
        #         "top_px": [int(top_px[0]), int(top_px[1])],
        #         "bottom_px": [int(bottom_px[0]), int(bottom_px[1])],
        #         "angle_deg": float(angle_deg),
        #     }
        #     result["cut_endpoints_world"] = {
        #         "top": [float(c) for c in top_inset],
        #         "bottom": [float(c) for c in bottom_inset],
        #     }
        return result

    # --- staging motions ------------------------------------------------------

    async def stop(self) -> Mapping[str, ValueTypes]:
        """Cancel whatever command is driving the arm, then halt the arm.

        Cancelling first means the interrupted sequence issues no further moves;
        the arm stop then halts the move already in flight. The arm is left
        where it stopped -- with a blade possibly in the tape, backing out is
        the operator's call.
        """
        task = self._task
        if task is not None:
            task.cancel()
        await self.arm.stop()
        return {"stopped": True, "interrupted": task.get_name() if task else None}

    async def home(self) -> Mapping[str, ValueTypes]:
        await self._move_home()
        return {"homed": True}

    async def _move_home(self) -> None:
        s = self.settings
        x, y, z = s.home_xyz
        await self._move(s.tool_frame, self._world_pose(x, y, z))

    async def jog(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        """Step the tool in its own frame and report the world pose either side.

        A bench tool for reading how the tool frame's axes land in world
        coordinates: `world_delta` is the answer, so checking an axis takes one
        call rather than a jog plus two pose reads.
        """
        s = self.settings
        step = {a: float(command.get(a) or 0.0) for a in ("x", "y", "z")}
        theta = float(command.get("theta") or 0.0)
        if not any(step.values()) and not theta:
            raise ValueError("jog needs a nonzero 'x', 'y', 'z' or 'theta'")
        too_far = [f"{a}={v}" for a, v in step.items() if abs(v) > JOG_MAX_MM]
        if too_far:
            raise ValueError(
                f"jog step exceeds JOG_MAX_MM={JOG_MAX_MM:.0f}: {', '.join(too_far)}"
            )

        before = await self.motion.get_pose(
            component_name=s.tool_frame, destination_frame=s.world_frame
        )
        await self._tool_move(theta=theta, **step)
        after = await self.motion.get_pose(
            component_name=s.tool_frame, destination_frame=s.world_frame
        )
        return {
            "jogged": {**step, "theta": theta},
            "world_before": _pose_to_dict(before.pose),
            "world_after": _pose_to_dict(after.pose),
            "world_delta": {
                a: round(float(getattr(after.pose, a) - getattr(before.pose, a)), 3)
                for a in ("x", "y", "z")
            },
        }

    async def tool_change(self) -> Mapping[str, ValueTypes]:
        await self._tool_change()
        return {"tool_changed": True}

    async def _tool_change(self) -> None:
        # Disabled until the tool-change deck is re-measured on the new arm: the
        # pose below is in the old arm's frame, so driving to it would be a guess.
        # s = self.settings
        # pose_above_deck = self._world_pose(251.86, -405.89, 115.23)
        # tool_change_descent = 121.5
        # await self.motion.move(component_name=s.tool_frame, destination=pose_above_deck)
        # # z goes to -6.75
        # tool_change_constraint=Constraints(
        #                linear_constraint=[
        #                    LinearConstraint(line_tolerance_mm=s.descent_tolerance_mm)
        #                ]
        #            )
        # await self.motion.move(component_name=s.tool_frame, destination=self._tool_pose(x=0, y=0, z=tool_change_descent), constraints=tool_change_constraint)
        # await self.motion.move(component_name=s.tool_frame, destination=self._tool_pose(x=0, y=0, z=-tool_change_descent), constraints=tool_change_constraint)
        return


    async def move_to_center(self) -> Mapping[str, ValueTypes]:
        """Run find_center, then descend to a standoff above the box top.

        Returns the find_center payload plus a `moved` key (the accumulating dict
        is intentional).
        """
        result = await self.find_center()
        if not result.get("found"):
            return result
        s = self.settings
        box = self._box_data
        await self._move(
            s.tool_frame,
            self._tool_pose(
                x=box.tool_x_mm,
                y=box.tool_y_mm,
                z=box.knife_tip_to_top_mm - s.center_standoff_mm,
            ),
            constraints=Constraints(
                linear_constraint=[
                    LinearConstraint(line_tolerance_mm=s.descent_tolerance_mm)
                ]
            ),
        )
        result["moved"] = True
        return result

    async def move_to_seam(self, seam: str | None) -> Mapping[str, ValueTypes]:
        """Run find_center, then descend to a standoff above the box top.

        Returns the find_center payload plus a `moved` key (the accumulating dict
        is intentional).
        """

        if seam is None:
            return {"error": "seam is required"}

        result = {}

        if seam == SEAM_FAR or seam == SEAM_CLOSE:
            await self._stage_side_seam(seam, self._box_data)
            result["moved"] = True
        else:
            result["error"] = "can only move to far or close seam"
        return result

    async def _stage_side_seam(self, seam: str, box: BoxData) -> None:
        """Park the blade at a side seam, angled and offset onto the tape.

        The far seam sits a box-height beyond the stopper; the close one sits at
        it. The blade tilts opposite ways for the two -- they are approached from
        opposite sides of the box -- and only the far approach takes the lateral
        nudge.
        """
        s = self.settings
        if seam == SEAM_FAR:
            # The box extends away from the base along +x, so the far edge sits a
            # box-span past the stopper rather than before it.
            seam_x = s.stopper_x_mm + box.height_mm
            blade_theta = -s.blade_angle_deg
            approach_lateral = FAR_SEAM_APPROACH_LATERAL_MM
        else:
            seam_x = s.stopper_x_mm
            blade_theta = s.blade_angle_deg
            approach_lateral = 0.0

        await self._move(
            s.tool_frame,
            self._world_pose(
                x=seam_x,
                y=box.center_y_mm,
                z=box.center_z_mm + s.side_seam_z_offset_mm,
                theta=SIDE_SEAM_THETA_DEG,
            ),
        )
        await self._move(s.blade_frame, self._blade_pose(theta=blade_theta))
        await self._tool_move(
            # Back off along the seam, so the stroke that follows cuts
            # through its whole length rather than starting mid-tape.
            x=-CUT_SIGN * s.seam_offset_fraction * box.flap_width_mm,
            y=approach_lateral,
        )

    # --- converge -------------------------------------------------------------

    async def converge(self, seam: str = SEAM_TOP) -> Mapping[str, ValueTypes]:
        """Visual-servo the blade onto a seam.

        The loop carries its own bounds, and a seam that will not
        converge comes back as `success: false` -- an outcome to branch on, not
        an error: the caller can retry the search without having cut anything.
        """
        s = self.settings
        box = self._box_data
        if box is None:
            return self._converge_failure(
                seam, 0, "no box frame; run find_center first"
            )

        if seam in (SEAM_FAR, SEAM_CLOSE):
            if not s.blade_frame:
                return self._converge_failure(
                    seam,
                    0,
                    "'blade_frame' is not configured; the side seams are cut with "
                    "the angled blade frame",
                )
            if box.flap_width_mm is None:
                return self._converge_failure(
                    seam,
                    0,
                    "no flap width for this box; detection cannot recover it, so "
                    "run set_box first",
                )

        inv_jacobian = invert_jacobian(s.jacobian_rows())
        gain = s.gain_for(seam)

        delta = (0.0, 0.0)
        iterations = 0
        blank_frames = 0
        error_px = None
        while iterations < s.converge_max_iterations:
            # A zero step is the script's no-op first move; skip the round trip.
            if delta != (0.0, 0.0):
                await self._tool_move(x=delta[0], y=delta[1])
            iterations += 1

            images, _ = await self.camera.get_images()
            color_bgr = decode_color(images)
            found = find_vertical_seam_line(
                color_bgr,
                blade_x_px=s.blade_x_px,
                search_radius_px=s.seam_search_radius_px,
            )
            if found is None:
                # If a seam isn't found, we count it as a "blank_frame". After
                # reaching 5 "blank_frames" in a row, the attempt stops with a fail.
                blank_frames += 1
                delta = (0.0, 0.0)
                if blank_frames >= s.converge_max_blank_frames:
                    return self._converge_failure(
                        seam,
                        iterations,
                        f"no seam line visible for {blank_frames} consecutive "
                        f"frames (converge_max_blank_frames="
                        f"{s.converge_max_blank_frames})",
                    )
                continue

            blank_frames = 0
            center_x = found[0]
            error_px = float(center_x - s.blade_x_px)
            if abs(error_px) <= s.converge_tolerance_px:
                return {
                    "success": True,
                    "seam": seam,
                    "iterations": iterations,
                    "error_px": error_px,
                }
            # Only the x error is servoed: the seam runs along the camera's
            # vertical, so its row tells us nothing about the blade's offset.
            delta = pixel_error_to_delta_mm((error_px, 0.0), inv_jacobian, gain)

        return self._converge_failure(
            seam,
            iterations,
            f"seam still {error_px:.1f}px from the blade after "
            f"converge_max_iterations={s.converge_max_iterations}"
            if error_px is not None
            else f"gave up after converge_max_iterations="
            f"{s.converge_max_iterations}",
            error_px=error_px,
        )

    @staticmethod
    def _converge_failure(seam, iterations, reason, error_px=None):
        return {
            "success": False,
            "seam": seam,
            "iterations": iterations,
            "error_px": error_px,
            "reason": reason,
        }

    # --- cut ------------------------------------------------------------------

    async def cut(self, seam: Optional[str] = None) -> Mapping[str, ValueTypes]:
        """Cut one seam. Committed motion: the blade is already on the seam.

        With no `seam`, the service works out which one it is parked at from the
        tool's world Y.
        """
        s = self.settings
        box = self._box_data
        if box is None:
            return {
                "completed": False,
                "seam": seam,
                "reason": "no box frame; run find_center first",
            }

        if seam is None:
            seam, reason = await self._seam_from_tool_pose(box)
            if seam is None:
                return {"completed": False, "seam": None, "reason": reason}

        if seam in (SEAM_FAR, SEAM_CLOSE) and not s.blade_frame:
            return {
                "completed": False,
                "seam": seam,
                "reason": "'blade_frame' is not configured; the side seams are cut "
                "with the angled blade frame",
            }

        if seam == SEAM_TOP:
            steps = await self._cut_top_seam(box)
        else:
            steps = await self._cut_side_seam(seam)
        return {"completed": True, "seam": seam, "steps": steps}

    async def _seam_from_tool_pose(self, box: BoxData):
        """Which seam is the blade parked at? Returns (seam, reason)."""
        s = self.settings
        pose = await self.motion.get_pose(
            component_name=s.tool_frame, destination_frame=s.world_frame
        )
        tool_x = float(pose.pose.x)
        candidates = sorted(
            (
                (abs(tool_x - box.center_x_mm), SEAM_TOP),
                (abs(tool_x - (s.stopper_x_mm + box.height_mm)), SEAM_FAR),
                (abs(tool_x - s.stopper_x_mm), SEAM_CLOSE),
            )
        )
        in_range = [c for c in candidates if c[0] <= s.seam_match_tolerance_mm]
        if not in_range:
            distance, nearest = candidates[0]
            return None, (
                f"nearest seam ({nearest}) is {distance:.0f}mm from the tool at "
                f"x={tool_x:.0f}, beyond seam_match_tolerance_mm="
                f"{s.seam_match_tolerance_mm:.0f}; converge first or pass 'seam'"
            )
        if len(in_range) > 1 and in_range[1][0] - in_range[0][0] < SEAM_AMBIGUITY_MM:
            return None, (
                f"cannot tell the {in_range[0][1]} seam from the {in_range[1][1]} "
                f"seam at x={tool_x:.0f} ({in_range[0][0]:.0f}mm vs "
                f"{in_range[1][0]:.0f}mm); pass an explicit 'seam'"
            )
        return in_range[0][1], None

    async def _cut_top_seam(self, box: BoxData) -> list:
        """Two passes from the center: forward, back to center, then backward.

        The blade was making an arch movement when cutting the top seam in one move
        , so each half is sliced from the middle outward. The slice
        itself is broken into chunks. Can be changed to one big cut with "LinearConstraint"
        """
        s = self.settings
        steps = []
        chunks = [f * box.height_mm for f in s.top_seam_chunks]
        cut_distance = sum(chunks)

        # TODO: double check this extra business
        await self._tool_move(z=s.top_blade_insert_mm)
        steps.append("insert")
        for chunk in chunks:
            await self._tool_move(x=CUT_SIGN * chunk)
            steps.append("slice_forward")
        await self._tool_move(z=-s.top_blade_insert_mm)
        steps.append("retract")

        await self._tool_move(x=-CUT_SIGN * cut_distance)
        steps.append("return_to_center")

        await self._tool_move(z=s.top_blade_insert_mm)
        steps.append("insert")
        for chunk in chunks:
            await self._tool_move(x=-CUT_SIGN * chunk)
            steps.append("slice_back")
        await self._tool_move(z=-(s.top_blade_insert_mm))
        steps.append("retract")
        return steps

    async def _cut_side_seam(self, seam: str) -> list:
        """Insert, one constrained stroke along the seam, retract, unwind."""
        s = self.settings
        if seam == SEAM_FAR:
            insert_z = s.side_blade_insert_mm
            retract_z = -s.side_blade_insert_mm
            straighten_theta = s.blade_angle_deg
        else:
            insert_z = s.side_blade_insert_mm
            # The close seam pulls far clear of the box on the way out, not just
            # back out of the tape.
            retract_z = -CLOSE_SEAM_RETRACT_MM
            straighten_theta = -s.blade_angle_deg

        steps = []
        await self._tool_move(z=insert_z)
        steps.append("insert")
        await self._tool_move(
            x=CUT_SIGN * s.side_seam_slice_mm,
            constraints=Constraints(
                linear_constraint=[
                    LinearConstraint(line_tolerance_mm=s.cut_tolerance_mm)
                ]
            ),
        )
        steps.append("slice")
        await self._tool_move(z=retract_z)
        steps.append("retract")
        await self._move(s.blade_frame, self._blade_pose(theta=straighten_theta))
        steps.append("straighten_blade")
        if seam == SEAM_CLOSE:
            await self._tool_move(theta=CLOSE_SEAM_FINAL_THETA_DEG)
            steps.append("straighten_tool")
        return steps

    # --- full_cut -------------------------------------------------------------

    async def full_cut(self) -> Mapping[str, ValueTypes]:
        """Home, find the box, then converge on and cut all three seams.

        Stops at the first failure and reports where: the arm is left where it
        stopped, since backing out of a half-cut seam is not something the
        service can decide on its own.
        """
        steps = []
        await self._move_home()
        steps.append("home")

        center = await self.move_to_center()
        if not center.get("found"):
            return {**center, "completed": False, "steps": steps}
        steps.append("move_to_center")

        seams = []
        for seam in SEAMS:
            if seam != SEAM_TOP:
                await self.move_to_seam(seam)
            converged = await self.converge(seam)
            if not converged.get("success"):
                return {
                    "completed": False,
                    "steps": steps,
                    "seams": seams,
                    "failed_at": seam,
                    "stage": "converge",
                    "reason": converged.get("reason"),
                }
            steps.append(f"converge_{seam}")

            # The sequence knows which seam it is on, so it never asks the
            # service to work that out from the pose.
            cut = await self.cut(seam)
            if not cut.get("completed"):
                return {
                    "completed": False,
                    "steps": steps,
                    "seams": seams,
                    "failed_at": seam,
                    "stage": "cut",
                    "reason": cut.get("reason"),
                }
            steps.append(f"cut_{seam}")
            seams.append(
                {
                    "seam": seam,
                    "iterations": converged["iterations"],
                    "error_px": converged.get("error_px"),
                    "steps": cut["steps"],
                }
            )

        await self._move_home()
        steps.append("home")
        return {
            "completed": True,
            "steps": steps,
            "seams": seams,
            "box_frame": center.get("box_frame"),
        }

    # --- pose helpers ---------------------------------------------------------

    def _world_pose(self, x, y, z, theta: float = 0.0) -> PoseInFrame:
        return PoseInFrame(
            reference_frame=self.settings.world_frame,
            pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=theta),
        )

    def _tool_pose(self, x=0.0, y=0.0, z=0.0, theta: float = 0.0) -> PoseInFrame:
        """A pose in the tool's own frame, i.e. a move relative to where it is."""
        return PoseInFrame(
            reference_frame=self.settings.tool_frame,
            pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=theta),
        )

    def _blade_pose(self, theta: float) -> PoseInFrame:
        return PoseInFrame(
            reference_frame=self.settings.blade_frame,
            pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=theta),
        )

    async def _move(self, component_name, destination, constraints=None):
        """Every arm move goes through here, so a dry run can gate them all."""
        await self.motion.move(
            component_name=component_name,
            destination=destination,
            constraints=constraints,
        )

    async def _tool_move(self, x=0.0, y=0.0, z=0.0, theta=0.0, constraints=None):
        await self._move(
            self.settings.tool_frame,
            self._tool_pose(x=x, y=y, z=z, theta=theta),
            constraints=constraints,
        )

    async def _endpoint_world(self, px, z, intr):
        ex, ey, ez = deproject(px[0], px[1], z, intr)
        pif = await self._to_frame((ex, ey, ez), self.settings.world_frame)
        return (pif.pose.x, pif.pose.y, pif.pose.z)

    async def _to_frame(self, point_xyz, dest_frame):
        """Transform a camera-frame point (mm) into dest_frame."""
        if not self.robot_client:
            self.robot_client = await create_robot_client_from_module()

        x, y, z = point_xyz
        observer_pose = PoseInFrame(
            reference_frame=self.settings.camera_frame,
            pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0),
        )
        return await self.robot_client.transform_pose(observer_pose, dest_frame)

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`get_status` is not implemented")
        raise NotImplementedError()

    async def close(self):
        if self._task is not None:
            self._task.cancel()
        if self.robot_client:
            await self.robot_client.close()
            self.robot_client = None
