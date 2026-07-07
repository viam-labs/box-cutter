# Box Cutter `control` Service — Design

Date: 2026-07-07

## Goal

Port the box-center detection and cut-motion-planning logic from the standalone
client script `../box-point-test/main.py` into the `viam-labs:box-cutter:control`
Generic service so it runs *on* the machine and is triggered via `do_command`.

Dropped from the original script: the interactive confirmation gate, debug-image
saving/annotation, and the OS image viewer.

## Key constraint discovered

The script uses `RobotClient.transform_pose(...)` to convert the deprojected
camera-frame point into `world` / tool-frame coordinates. That call lives **only**
on `RobotClient` (`viam/robot/client.py`), not on any resource dependency a module
receives. The motion service exposes `get_pose(component_name, destination_frame,
supplemental_transforms=...)` instead.

Replacement: inject the deprojected point as a temporary frame attached to the
camera frame, then ask the motion service for its pose in the destination frame.

```python
t = Transform(
    reference_frame="box_point",
    pose_in_observer_frame=PoseInFrame(
        reference_frame=camera_frame,     # e.g. "realsense-cam"
        pose=Pose(x=x, y=y, z=z, o_z=1),
    ),
)
world_pif = await motion.get_pose("box_point", world_frame, supplemental_transforms=[t])
```

This produces identical results to `transform_pose` without a `RobotClient`.

## File layout

- `src/models/detection.py` — ported **pure** vision/geometry functions (no I/O,
  unit-testable): `detect_box_center` (debug `imwrite` removed), `find_seam_edges`,
  `_pick_seam_axis`, `inset_endpoints`, `sample_depth_in_mask`, `deproject`,
  `decode_color_and_depth`. Tuning constants become function parameters (the
  resource passes configured values; module-level constants remain as defaults).
- `src/models/control.py` — the `Control` resource: config parse/validate,
  dependency wiring, frame-transform helper, and the three action methods behind a
  `do_command` dispatch.
- `requirements.txt` — add `opencv-python` and `numpy`.

## Configuration

Read in `validate_config` (types checked, required deps returned) and applied in
`reconfigure`.

Resource names (returned as **required dependencies**):
- `camera` (required) — the RealSense camera resource name.
- `arm` (required) — the arm component resource name.
- `motion_service` (optional, default `"builtin"`).

Frame names:
- `camera_frame` (optional, defaults to the `camera` name).
- `tool_frame` (required) — e.g. `"stylus-tool"`.
- `world_frame` (optional, default `"world"`).

Optional tuning (defaults = script constants):
- Detection: `hsv_lower` [10,60,80], `hsv_upper` [35,255,255], `min_box_area` 5000,
  `inset_mm` 8, `min_seam_len_px` 60, `seam_dark_v_max` 80.
- Motion: `plunge_depth_mm` 4, `slice_depth_mm` 3, `lift_mm` 100,
  `twist_joint_index` 5, `twist_angle_deg` 180, `line_tolerance_mm` 1.

## `do_command` interface

Wire format: `{"command": "<name>", ...optional overrides}`. `do_command` validates
the `command` key and dispatches to a typed async method; unknown commands raise
`ValueError`.

### `find_center` — detection/planning only (no motion)
1. `camera.get_images()` → `decode_color_and_depth` (raise if color or depth frame
   missing; raise if depth resolution != color resolution).
2. `detect_box_center` → `(u, v, mask)`; if none, return `{found: false, reason}`.
3. `sample_depth_in_mask` → median masked depth `z`; if none, `{found: false, reason}`.
4. `deproject(u, v, z, intrinsics)` → camera-frame point; transform to `world`.
5. `find_seam_edges` → `top_px`, `bottom_px`, `angle_deg`; transform both endpoints
   to `world`, then `inset_endpoints`.

Returns:
```json
{
  "found": true,
  "u": 0, "v": 0, "depth_mm": 0.0,
  "camera_frame_xyz": {"x":0,"y":0,"z":0},
  "world_pose": {"x":0,"y":0,"z":0,"o_x":0,"o_y":0,"o_z":0,"theta":0},
  "seam": {"top_px":[0,0], "bottom_px":[0,0], "angle_deg":0.0},
  "cut_endpoints_world": {"top":[0,0,0], "bottom":[0,0,0]}
}
```

### `move_to_center`
Runs `find_center`; if a box was found, `motion.move(tool_frame, center)` where the
goal is the world center with `z -= plunge_depth_mm` and orientation pointing down
(`o_z = -1, theta = 0`). Returns the `find_center` payload + `{"moved": true}`.

### `full_cut`
Runs `move_to_center`, then the ported slice sequence (from the commented block in
the script), parameterized by the motion tuning attributes:
1. Capture home pose: `motion.get_pose(arm, world_frame)` at the start.
2. Plunge to `bottom_inset` (`z -= slice_depth_mm`).
3. Lift `+lift_mm`.
4. Twist: read arm joints, set `joints[twist_joint_index] = twist_angle_deg`,
   `arm.move_to_joint_positions`.
5. Linear-constrained slice to `top_inset` (`LinearConstraint(line_tolerance_mm)`).
6. Lift `+lift_mm`.
7. Untwist joint back to 0.
8. Return arm to captured home pose.

Returns `{"completed": true, "steps": [...]}`.

## Error handling

- Detection misses (`no box` / `no valid depth`) → structured `{found: false,
  reason}` so callers can branch without exceptions.
- Config errors, missing camera frames, unaligned depth, hardware failures → raise
  `ValueError` / propagate.
- All return values coerced to JSON-native types (numpy scalars → `float`/`int`) so
  they are valid `ValueTypes`.

## Testing

- Unit tests for `detection.py` pure functions:
  - `deproject` — known intrinsics/inputs → expected mm.
  - `inset_endpoints` — normal span and degenerate (span < 2·inset) collapse.
  - `detect_box_center` — synthetic tan rectangle on neutral background → center
    near geometric center; sub-threshold speck → `None`.
  - `sample_depth_in_mask` — masked median ignoring zeros.
- Motion paths (`move_to_center`, `full_cut`) verified manually against a live
  machine; they require real hardware and the frame system.
- `get_status` remains the existing stub (not part of this work).
