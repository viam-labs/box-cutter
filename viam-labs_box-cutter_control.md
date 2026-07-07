# Model viam-labs:box-cutter:control

A generic service that locates a tan cardboard box in a RealSense color+depth
frame and plans/executes a cutting motion for a robot arm.

It segments the box by HSV color, finds the box centroid, samples the box depth,
deprojects the center (and the tape-seam cut endpoints) into 3D, and transforms
those points into the robot's `world` frame via the motion service. Depending on
the command, it can just report the geometry, move the tool to the box center, or
run the full cut sequence.

## Requirements

The machine must be configured with:

- A depth camera (e.g. RealSense) that returns an aligned color (JPEG/PNG) frame
  **and** a `VIAM_RAW_DEPTH` frame from `GetImages`, and exposes intrinsic
  parameters. Depth must be aligned to color (same resolution).
- An arm component.
- The built-in motion service (used for frame transforms and moves).

## Configuration

The following attribute template can be used to configure this model:

```json
{
  "camera": "realsense-cam",
  "arm": "xarm",
  "tool_frame": "stylus-tool",
  "motion_service": "builtin",
  "camera_frame": "realsense-cam",
  "world_frame": "world"
}
```

### Attributes

| Name                | Type            | Inclusion | Default            | Description                                                                 |
|---------------------|-----------------|-----------|--------------------|-----------------------------------------------------------------------------|
| `camera`            | string          | Required  | —                  | Resource name of the depth camera. Added as a dependency.                   |
| `arm`               | string          | Required  | —                  | Resource name of the arm. Added as a dependency.                            |
| `tool_frame`        | string          | Required  | —                  | Frame name of the cutting tool to move (e.g. `stylus-tool`).                |
| `motion_service`    | string          | Optional  | `builtin`          | Resource name of the motion service. Added as a dependency.                 |
| `camera_frame`      | string          | Optional  | value of `camera`  | Frame name of the camera in the frame system.                               |
| `world_frame`       | string          | Optional  | `world`            | Destination reference frame for computed poses and moves.                   |
| `hsv_lower`         | [int, int, int] | Optional  | `[10, 60, 80]`     | Lower HSV bound for box color segmentation.                                 |
| `hsv_upper`         | [int, int, int] | Optional  | `[35, 255, 255]`   | Upper HSV bound for box color segmentation.                                 |
| `min_box_area`      | int             | Optional  | `5000`             | Minimum contour area (px²) to accept a detection.                           |
| `inset_mm`          | float           | Optional  | `8`                | Distance each cut endpoint is pulled inward from the box edge.              |
| `min_seam_len_px`   | int             | Optional  | `60`               | Minimum plausible seam length (px); shorter seams are rejected.             |
| `seam_dark_v_max`   | int             | Optional  | `80`               | HSV value below which a box pixel counts as part of the (dark) tape seam.   |
| `plunge_depth_mm`   | float           | Optional  | `4`                | How far below the detected center Z the tool descends on `move_to_center`.  |
| `slice_depth_mm`    | float           | Optional  | `3`                | How far below the cut plane the tool slices during `full_cut`.              |
| `lift_mm`           | float           | Optional  | `100`              | Lift height between cut passes during `full_cut`.                           |
| `twist_joint_index` | int             | Optional  | `5`                | Arm joint index rotated during `full_cut`.                                  |
| `twist_angle_deg`   | float           | Optional  | `180`              | Angle (degrees) the twist joint is rotated to, then back to 0.             |
| `line_tolerance_mm` | float           | Optional  | `1`                | Linear-move tolerance for the constrained slice pass.                       |

### Example Configuration

```json
{
  "camera": "realsense-cam",
  "arm": "xarm",
  "tool_frame": "stylus-tool"
}
```

## DoCommand

All actions are triggered through `DoCommand` with a `command` key naming the
action. An unknown or missing `command` raises an error.

### `find_center`

Detect the box and report its geometry. Performs **no motion**.

```json
{ "command": "find_center" }
```

Returns (when a box is found):

```json
{
  "found": true,
  "u": 320,
  "v": 240,
  "depth_mm": 700.0,
  "camera_frame_xyz": { "x": 0.0, "y": 0.0, "z": 700.0 },
  "world_pose": { "x": 0, "y": 0, "z": 0, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0 },
  "seam": { "top_px": [320, 300], "bottom_px": [320, 180], "angle_deg": 90.0 },
  "cut_endpoints_world": { "top": [0, 0, 0], "bottom": [0, 0, 0] }
}
```

If no box (or no valid depth) is found, returns `{ "found": false, "reason": "..." }`.
The `seam` / `cut_endpoints_world` keys are present only when a seam is detected.

### `move_to_center`

Run `find_center`, then move the tool to the box center (descending
`plunge_depth_mm` below the detected surface). Returns the `find_center` payload
plus `"moved": true`. If no box is found, returns the `find_center` result and
performs no motion.

```json
{ "command": "move_to_center" }
```

### `full_cut`

Run `move_to_center`, then execute the full cut sequence (plunge, lift, twist,
linear-constrained slice, lift, untwist, return to the arm's home pose). Returns
the `move_to_center` payload plus `"completed": true` and a `"steps"` list. If no
box is found it aborts before moving; if a box is found but no seam/cut endpoints
are detected it moves to center and returns `"completed": false` with a reason.

```json
{ "command": "full_cut" }
```
