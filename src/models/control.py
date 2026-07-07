from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, Transform
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Constraints, MotionClient
from viam.proto.service.motion import LinearConstraint
from viam.utils import ValueTypes

from models.detection import (
    decode_color_and_depth,
    deproject,
    detect_box_center,
    find_seam_edges,
    inset_endpoints,
    sample_depth_in_mask,
)


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


def _pose_to_dict(pose) -> dict:
    return {
        "x": float(pose.x), "y": float(pose.y), "z": float(pose.z),
        "o_x": float(pose.o_x), "o_y": float(pose.o_y), "o_z": float(pose.o_z),
        "theta": float(pose.theta),
    }


@dataclass
class Settings:
    camera_name: str
    arm_name: str
    tool_frame: str
    motion_name: str
    camera_frame: str
    world_frame: str
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    min_box_area: int
    inset_mm: float
    min_seam_len_px: int
    seam_dark_v_max: int
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
        return super().new(config, dependencies)

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
        if not intr.focal_x_px or not intr.focal_y_px:
            raise ValueError(
                "camera returned no intrinsic parameters; cannot deproject "
                "(configure the camera to emit intrinsics)"
            )

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

    async def move_to_center(self) -> Mapping[str, ValueTypes]:
        """Run find_center, then move the tool to the center; returns the
        find_center payload plus a `moved` key (the accumulating dict is intentional)."""
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

    async def full_cut(self) -> Mapping[str, ValueTypes]:
        """Run move_to_center, then execute the slice sequence; returns the
        move_to_center payload plus `completed`/`steps` (the accumulating dict is intentional)."""
        s = self.settings
        home = await self.motion.get_pose(
            component_name=s.arm_name, destination_frame=s.world_frame
        )

        # Validate the twist index before any motion, so we never park the tool
        # mid-cut (post-plunge) only to throw on an out-of-range list index.
        joints = await self.arm.get_joint_positions()
        if not (0 <= s.twist_joint_index < len(joints.values)):
            raise ValueError(
                f"twist_joint_index {s.twist_joint_index} out of range for arm with "
                f"{len(joints.values)} joints"
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
        cut_z = bottom[2] - s.slice_depth_mm
        lift_z = cut_z + s.lift_mm

        def down_pose(x, y, z):
            return PoseInFrame(
                reference_frame=s.world_frame,
                pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0),
            )

        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(bottom[0], bottom[1], cut_z),
        )
        steps.append("plunge_bottom")

        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(bottom[0], bottom[1], lift_z),
        )
        steps.append("lift")

        joints = await self.arm.get_joint_positions()
        joints.values[s.twist_joint_index] = s.twist_angle_deg
        await self.arm.move_to_joint_positions(joints)
        steps.append("twist")

        # Slice rides the same cut plane as the plunge (cut_z, derived from
        # bottom[2]), not top[2] -- the seam is cut at a single consistent depth.
        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(top[0], top[1], cut_z),
            constraints=Constraints(
                linear_constraint=[LinearConstraint(line_tolerance_mm=s.line_tolerance_mm)]
            ),
        )
        steps.append("slice_top")

        await self.motion.move(
            component_name=s.tool_frame,
            destination=down_pose(top[0], top[1], lift_z),
        )
        steps.append("lift")

        joints = await self.arm.get_joint_positions()
        joints.values[s.twist_joint_index] = 0
        await self.arm.move_to_joint_positions(joints)
        steps.append("untwist")

        await self.motion.move(component_name=s.arm_name, destination=home)
        steps.append("home")

        result["completed"] = True
        result["steps"] = steps
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

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`get_status` is not implemented")
        raise NotImplementedError()
