from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import MotionClient
from viam.utils import ValueTypes


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
        self.camera = self._resolve(
            dependencies, Camera.get_resource_name(settings.camera_name)
        )
        self.arm = self._resolve(
            dependencies, Arm.get_resource_name(settings.arm_name)
        )
        self.motion = self._resolve(
            dependencies, MotionClient.get_resource_name(settings.motion_name)
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`get_status` is not implemented")
        raise NotImplementedError()
