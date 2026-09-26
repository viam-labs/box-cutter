# Model viam-labs:box-cutter:control

A generic service that locates a cardboard box in a RealSense color+depth frame
and cuts its three tape seams with a blade on a robot arm: the seam across the
box top, the side seam furthest from the robot base, and the side seam nearest
it.

Each seam is cut in two stages. `converge` positions the blade and then visually
servos it onto the seam line, reporting whether it got there; `cut` runs the
slicing motion for whichever seam the blade is sitting at. `full_cut` chains all
of it together.

## Requirements

The machine must be configured with:

- A depth camera (e.g. RealSense) that returns an aligned color (JPEG/PNG) frame
  **and** a `VIAM_RAW_DEPTH` frame from `GetImages`, and exposes intrinsic
  parameters. Depth must be aligned to color (same resolution).
- An arm component.
- The built-in motion service.
- A frame for the cutting tool (`tool_frame`), and — for the side seams — a
  second frame for the angled blade (`blade_frame`).
- `VIAM_API_KEY`, `VIAM_API_KEY_ID`, and `VIAM_MACHINE_FQDN` in the module's
  environment; the service opens a robot client to transform camera points into
  the world and tool frames.

## Configuration

```json
{
  "camera": "realsense-cam",
  "arm": "arm-UR5",
  "tool_frame": "stylus-tool",
  "blade_frame": "blade-tool-90deg",
  "motion_service": "builtin",
  "camera_frame": "realsense-cam",
  "world_frame": "world"
}
```

### Attributes

| Name | Type | Inclusion | Default | Description |
|---|---|---|---|---|
| `camera` | string | Required | — | Resource name of the depth camera. Added as a dependency. |
| `arm` | string | Required | — | Resource name of the arm. Added as a dependency. |
| `tool_frame` | string | Required | — | Frame of the cutting tool (e.g. `stylus-tool`). |
| `blade_frame` | string | Optional | `""` | Frame used to tilt the blade for side seams (e.g. `blade-tool-90deg`). Side-seam commands fail without it. |
| `motion_service` | string | Optional | `builtin` | Resource name of the motion service. Added as a dependency. |
| `camera_frame` | string | Optional | value of `camera` | Frame name of the camera in the frame system. |
| `world_frame` | string | Optional | `world` | Destination reference frame for computed poses. |

#### Box detection

| Name | Type | Inclusion | Default | Description |
|---|---|---|---|---|
| `hsv_lower` | [int, int, int] | Optional | `[10, 60, 80]` | Lower HSV bound for box color segmentation. |
| `hsv_upper` | [int, int, int] | Optional | `[35, 255, 255]` | Upper HSV bound for box color segmentation. |
| `min_box_area` | int | Optional | `5000` | Minimum contour area (px²) to accept a detection. |
| `inset_mm` | float | Optional | `8` | Inset applied to the reported seam endpoints. |
| `min_seam_len_px` | int | Optional | `60` | Minimum plausible seam length (px). |
| `seam_dark_v_max` | int | Optional | `80` | HSV value below which a box pixel counts as tape. |

#### Cell geometry (measured)

| Name | Type | Inclusion | Default | Description |
|---|---|---|---|---|
| `home_xyz` | [float × 3] | Optional | `[399.97, 0, 406.48]` | Home pose of the tool in the world frame; blade vertical, pointing down. |
| `stopper_x_mm` | float | Optional | `332` | World X of the physical box stopper, which the box's close edge rests against. |
| `knife_tip_to_table_mm` | float | Optional | `435` | Knife tip height above the table at arm home. |
| `base_plate_height_mm` | float | Optional | `20` | Height of the plate the arm is bolted to. |

#### Visual servoing

| Name | Type | Inclusion | Default | Description |
|---|---|---|---|---|
| `blade_x_px` | float | Optional | `344` | Pixel column the blade occupies in the camera frame. |
| `servo_jacobian` | [float × 4] | Optional | `[0, 3.2, -3.2, 0]` | Image Jacobian `[du/dX, du/dY, dv/dX, dv/dY]`. |
| `seam_search_radius_px` | int | Optional | `40` | Ignore seam lines further than this from the blade column. |
| `converge_tolerance_px` | float | Optional | `2.25` | Pixel error at which a seam counts as converged. |
| `converge_max_iterations` | int | Optional | `25` | Servo iterations before giving up. |
| `converge_max_blank_frames` | int | Optional | `5` | Consecutive frames with no visible seam before giving up. |
| `top_seam_gain` | float | Optional | `0.2` | Proportional servo gain for the top seam. |
| `far_seam_gain` | float | Optional | `0.05` | Proportional servo gain for the far side seam. |
| `close_seam_gain` | float | Optional | `0.09` | Proportional servo gain for the close side seam. |

#### Cutting motion

| Name | Type | Inclusion | Default | Description |
|---|---|---|---|---|
| `center_standoff_mm` | float | Optional | `20` | How far above the box top the tool stops on `move_to_center`. |
| `top_blade_insert_mm` | float | Optional | `25` | Blade insertion depth for the top seam. |
| `side_blade_insert_mm` | float | Optional | `16` | Blade insertion depth for the side seams. The close seam inserts 7 mm deeper than this. Must not exceed 33, so the close-seam insert stays within its 40 mm retract. |
| `top_seam_chunks` | [float] | Optional | `[0.15, 0.15, 0.25]` | Top-seam pass split into these fractions of box height; their sum is the travel per direction. |
| `side_seam_slice_mm` | float | Optional | `65` | Slice distance along each side seam. |
| `blade_angle_deg` | float | Optional | `15` | Blade tilt applied before a side cut, and undone after. |
| `seam_offset_fraction` | float | Optional | `0.55` | Side-seam approach offset, as a fraction of flap width. |
| `side_seam_z_offset_mm` | float | Optional | `10` | Height above the box top for the side-seam approach. |
| `descent_tolerance_mm` | float | Optional | `10` | Linear tolerance for the descent to box center. |
| `cut_tolerance_mm` | float | Optional | `3` | Linear tolerance for a side-seam slice. |
| `seam_match_tolerance_mm` | float | Optional | `40` | How close the tool must be to a seam for `cut` to infer it. |

#### Dry run

Only used during a dry run, but validated at config time — an invalid
`workspace_min_xyz`/`workspace_max_xyz` or a negative `dry_run_clearance_mm`
blocks the config even if every run is a real one.

| Name | Type | Inclusion | Default | Description |
|---|---|---|---|---|
| `dry_run_clearance_mm` | float | Optional | `30` | Extra height added to every approach standoff during a dry run. Must not be negative. |
| `workspace_min_xyz` | [float × 3] | Optional | — | Lower corner of the allowed volume for the knife tip, world frame. Set together with `workspace_max_xyz`. |
| `workspace_max_xyz` | [float × 3] | Optional | — | Upper corner. Must not be below `workspace_min_xyz` on any axis. |

## DoCommand

All actions are triggered through `DoCommand` with a `command` key. An unknown or
missing `command` raises an error, as does an unknown `seam`.

### `stop`

Cancel the command currently driving the arm and halt the arm. The interrupted
command returns `{"stopped": true, "command": "<name>"}`; `stop` itself returns
`{"stopped": true, "interrupted": "<name>" | null}`. The arm is left where it
stopped.

Only one arm command runs at a time: sending another while one is in flight
raises a `busy` error. `stop` and `get_properties` are always accepted.

```json
{ "command": "stop" }
```

### Dry run

`home`, `jog`, `move_to_center`, `move_to_seam`, `converge`, `cut`, and
`full_cut` accept a `"dry_run": true` key. A non-boolean `dry_run` is rejected
on every command that reaches the busy check — only `stop` and
`get_properties` never look at it. A boolean `dry_run` on any other command
(e.g. `set_box`, `find_center`, `tool_change`) is accepted but ignored.

The arm runs the real sequence, but:

- blade inserts and retracts are skipped and recorded in `skipped`, in order,
  as `"<seam>:<insert|retract>"`;
- the close seam's retract, which is a real move (it pulls clear of the box,
  not just out of the tape), is shortened by the skipped insert depth;
- every approach standoff is raised by `dry_run_clearance_mm`;
- before the first move, `tool_frame`, `blade_frame`, `camera_frame`, and
  `world_frame` must all exist in the frame system;
- if `workspace_min_xyz`/`workspace_max_xyz` are configured, any move whose
  target would leave that volume is refused before it is sent.

`home` and `jog` have no approach standoff, so a dry run does not raise them:
a dry-run `jog` with `z` moves the blade exactly as a real one does, including
downward. "Blade held clear" does not apply to either — they are only frame-
and bounds-checked.

A real run (no `dry_run`, or `dry_run: false`) makes none of these checks and
issues no extra calls.

```json
{ "command": "full_cut", "dry_run": true }
```

The response is the normal result plus:

```json
{
  "dry_run": true,
  "skipped": ["top:insert", "top:retract", "top:insert", "top:retract", "far:insert", "far:retract", "close:insert"],
  "bounds_checked": true
}
```

A dry run interrupted by `stop` returns the same plain response as any other
interrupted command: `{"stopped": true, "command": "<name>"}` — no `dry_run`
fields are added.

**What the bounds check covers**

- Bounds are in the world frame and cover the tool origin (the knife tip)
  only — not the arm links, the camera, or the blade body. Blade rotations
  aren't checked.
- Only move *targets* are checked, not the planned path between them: a
  free-space move such as `home` or a side-seam approach can swing outside
  the workspace between its start and end pose without being caught.
- A dry run checks only the targets of the raised path. A real run's targets
  sit up to `dry_run_clearance_mm` plus the insert depth lower
  (`top_blade_insert_mm` on the top seam, `side_blade_insert_mm` on the side
  seams, plus 7 mm on the close seam), and more after `converge`, which
  settles differently at the real height. For a passing dry run to mean
  anything about the real one, set `workspace_min_xyz`'s z at least
  `dry_run_clearance_mm` + the deepest insert depth, plus a margin, **above**
  the lowest height the knife tip may safely reach.
- Without `workspace_min_xyz`/`workspace_max_xyz` configured, the dry run
  still runs, logs a warning, and returns `"bounds_checked": false`.
- The per-move bounds check costs one extra `transform_pose` call, and only
  when a workspace is configured and the move is tool-relative (not already
  in the world frame, and not a blade-frame rotation). Separately, the frame
  check on the first move of each dry run makes 4 `transform_pose` calls, one
  per configured frame.

**`converge` in a dry run** servos from the raised height. The servo Jacobian
and pixel settings were tuned at the real standoff, so `iterations` and
`error_px` won't match a real run exactly — use it to confirm direction, not
tuning.

**Commissioning a new box or cell:** run `full_cut` with `dry_run` first and
confirm every move goes the way you expect, then run it for real.

**Caveat:** a dry-run `cut` only stays clear of the box if the move that
positioned the arm was also a dry run (`full_cut`, or `move_to_center` /
`move_to_seam` with `dry_run`). After a real `move_to_center`, a dry-run top
cut slices at the real `center_standoff_mm` standoff, where a raised flap can
still be hit. After a real `move_to_seam`, a dry-run side cut slices at the
real approach height with only the insert skipped, so the blade can still
touch the box.

### `set_box`

Override the detected box with measured values. Detection finds *where* the box
is but not how wide its flaps are, and the depth read drifts on a glossy or
badly-lit top, so the side seams need this. The override survives until it is
cleared or the service is reconfigured.

```json
{ "command": "set_box", "depth_mm": 535, "u": 380, "v": 242, "flap_width_mm": 120 }
{ "command": "set_box", "preset": "box_1" }
{ "command": "set_box", "clear": true }
```

`depth_mm` is the camera-to-box-top distance, `u`/`v` the box center pixel, and
`flap_width_mm` the distance from the center of the top to its side. Presets
`box_1` … `box_5` hold the boxes measured on the original cell.

### `home`

Move the tool to `home_xyz`, blade vertical. Returns `{"homed": true, "pose": {...}}`.

### `find_center`

Detect the box, apply any `set_box` override, and derive the box frame. No motion.

```json
{
  "found": true,
  "u": 380, "v": 242, "depth_mm": 535.0,
  "override_applied": true,
  "camera_frame_xyz": { "x": 0.0, "y": 0.0, "z": 535.0 },
  "world_pose": { "x": 0, "y": 0, "z": 0, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0 },
  "box_frame": {
    "center_x_mm": 0.0, "center_y_mm": -350.0, "center_z_mm": 180.0,
    "height_mm": 200.0, "knife_tip_to_top_mm": 290.0, "flap_width_mm": 120.0
  },
  "seam": { "top_px": [320, 300], "bottom_px": [320, 180], "angle_deg": 90.0 },
  "cut_endpoints_world": { "top": [0, 0, 0], "bottom": [0, 0, 0] }
}
```

If no box (or no valid depth) is found, returns `{ "found": false, "reason": "..." }`.
`seam` / `cut_endpoints_world` are reported for inspection only — the cut sequence
servos onto the seam visually rather than driving to those endpoints.

### `move_to_center`

Run `find_center`, then descend in the tool frame to `center_standoff_mm` above
the box top. Returns the `find_center` payload plus `"moved": true`.

### `converge`

Line the blade up with a seam and report whether it got there. The top seam is
assumed to be under the blade already (via `move_to_center`); a side seam is
staged first — the tool moves to that seam's approach pose, the blade tilts by
`blade_angle_deg`, and it offsets by `seam_offset_fraction × flap_width_mm`.

```json
{ "command": "converge", "seam": "far" }
```

`seam` is `top` (default), `far`, or `close`. Returns:

```json
{ "success": true, "seam": "far", "iterations": 7, "error_px": 1.8 }
```

On failure, `success` is `false` with a `reason`: the servo ran out of
iterations, no seam line was visible near the blade, `find_center` has not run,
`blade_frame` is not configured, or the flap width is unknown (no `set_box`).

### `cut`

Run the cutting motion for the seam the blade is at. With no `seam` argument the
service infers it from the tool's world pose against the three approach positions
derived from the box frame; if the tool is not near any of them, or two are
equally close, it refuses and moves nothing.

```json
{ "command": "cut" }
{ "command": "cut", "seam": "close" }
```

Returns `{"completed": true, "seam": "top", "steps": [...]}`. The top seam is cut
outward from the center in both directions; a side seam is a single
linear-constrained slice, after which the blade is straightened.

### `full_cut`

`home` → `move_to_center` → `converge`/`cut` for the top, far, and close seams →
`home`. Stops at the first stage that does not succeed.

```json
{ "command": "full_cut" }
```

Returns `{"completed": true, "steps": [...], "seams": [...], "box_frame": {...}}`,
or `{"completed": false, "failed_at": "far", "stage": "converge", "reason": "..."}`.

Run `set_box` first: the side seams need a flap width, which detection cannot
supply.
