# Dry run for the box-cutter control service

Date: 2026-09-24
Status: approved design, not yet implemented
Builds on: PR #5 (`stop` command and one-command-at-a-time guard)

## Goal

Make it safe to commission the service on the new arm. A dry run moves the arm
through a full sequence for real, with the blade held clear of the box, so an
operator can confirm the direction and orientation of every move (`CUT_SIGN`,
`SIDE_SEAM_THETA_DEG`) before any contact.

## Non-goals

- Planning-only / report-only mode (no motion).
- Checking frames or bounds during real runs. A real run behaves exactly as it
  does today, with no extra calls.
- A "verified values" gate on side-seam commands.
- Fixing `tool_change` (it still reports success without moving; tracked
  separately).
- Pre-validating a whole sequence before it starts. `converge` is closed-loop,
  and predicting relative poses would need pose-composition math the Python SDK
  does not provide.

## Interface

`home`, `jog`, `move_to_center`, `move_to_seam`, `converge`, `cut` and
`full_cut` accept `"dry_run": true`. It is per command only; there is no config
default, so a dry run cannot be left on and make a real cut look like a dry one.

A dry-run response carries, in addition to the command's normal fields:

- `"dry_run": true`
- `"skipped"`: the blade insert and retract moves that were not sent, in order,
  across the whole command (so a dry-run `full_cut` lists all three seams')
- `"bounds_checked"`: `true` if workspace bounds were configured and checked,
  `false` otherwise

`do_command` stores the flag on the instance for the length of the one command
and resets it in a `finally`. The busy guard from PR #5 means only one command
can be setting it. The new per-command state (the flag, the `skipped` list, and
whether the frames have been checked) gets class-level defaults, like `_task`,
so an instance built without `reconfigure` (as the tests do) still works.

`dry_run` on any other command (`find_center`, `set_box`, `get_properties`,
`tool_change`, `stop`) is ignored: none of them move the arm. Their responses
get no dry-run fields. A dry-run command interrupted by `stop` returns the
usual `{"stopped": true, "command": ...}` with no dry-run fields either.

## Config

New attributes, read only during a dry run:

| Name | Type | Default | Purpose |
|---|---|---|---|
| `dry_run_clearance_mm` | float | `30` | Extra height added to every approach standoff |
| `workspace_min_xyz` | [float × 3] | unset | Lower corner of the allowed cell volume, world frame |
| `workspace_max_xyz` | [float × 3] | unset | Upper corner |

The two bounds are set together or not at all. Setting only one is a config
validation error, and so is any min axis greater than its max. If neither is
set, a dry run still runs, logs a warning, and returns
`"bounds_checked": false`.

## The gate

Every `motion.move` call in `control.py` goes through one method:
`_move(component_name, destination, constraints=None, plunge=None)`. The direct
calls today (`_move_home`, `move_to_center`, the three moves in
`_stage_side_seam`, the `converge` step, and the blade straighten in
`_cut_side_seam`) switch to `_move`. `_tool_move` stays as the helper for
tool-relative moves and calls `_move`, passing `plunge` through. `plunge` is
`None` for a normal move, or a label for a blade insert or retract. With
`dry_run` false, it calls `motion.move` directly and nothing else. With
`dry_run` true, it runs these steps in order:

1. **Frame check, once per command.** For each of `tool_frame`, `blade_frame`,
   `camera_frame` and `world_frame`, transform a zero pose in that frame to
   `world_frame` with `robot_client.transform_pose`. The first failure raises
   `ValueError` naming the frame. This is the same call `find_center` already
   makes, so it works however the frame is defined (component frame or extra
   transform). `_to_frame` hard-codes the camera frame as its source, so this
   needs its own small helper. It opens the robot client the same lazy way
   `_to_frame` does.
2. **Blade in/out skip.** If `plunge` is set, append its label to `skipped`
   and return without sending the move.
3. **Bounds check** (only when bounds are configured). A destination already in
   `world_frame` is checked directly. A tool- or blade-relative destination is
   first transformed to `world_frame` with `transform_pose`, which gives the
   exact world pose the move will reach. Any axis outside `[min, max]` raises
   `ValueError` with the target and the bounds, and nothing is sent.
4. Send the move with `motion.move`.

### Which moves are plunges

Labels are `"<seam>:<insert|retract>"`, and `skipped` lists them in the order
they would have run:

- Top seam: both inserts (`z=+top_blade_insert_mm`) and both retracts
  (`z=-top_blade_insert_mm`), giving
  `["top:insert", "top:retract", "top:insert", "top:retract"]`.
- Far seam: the insert and the retract, giving
  `["far:insert", "far:retract"]`.
- Close seam: the insert only, giving `["close:insert"]`. See below.

While `side_blade_insert_mm` stays at its work-in-progress default of `0`, the
side-seam insert is already a zero-length move, so skipping it removes nothing.
A side-seam dry run keeps the blade clear through the raised approach height
alone (see Raised standoffs).

Insert and retract are skipped as a pair so the tool does not creep upward by
the insert depth on every pass.

### Close-seam retract in a dry run

In a real run, the close seam retracts `CLOSE_SEAM_RETRACT_MM` (40 mm), which
is more than it inserted, to pull clear of the box before the tool turns. In a
dry run the insert is skipped, and the retract becomes
`-(CLOSE_SEAM_RETRACT_MM - side_blade_insert_mm)`. The tool ends up the same
distance above its approach height as in a real run. The dry-run approach is
itself `dry_run_clearance_mm` higher, so the tool turns that much higher than in
a real run, which only adds margin. The retract goes through the gate as a
normal move and gets bounds-checked.

### Raised standoffs

These are two edits where the sequences compute heights, not part of the gate:

- `move_to_center` descends to `center_standoff_mm + dry_run_clearance_mm`
  above the box top.
- `_stage_side_seam` approaches at
  `center_z_mm + side_seam_z_offset_mm + dry_run_clearance_mm`.

`converge` runs its full vision loop. Its sideways steps are real moves, so
they get bounds-checked. At the raised height the seam may look different to
the camera; if `converge` fails in a dry run for that reason, that is useful
to know too.

### Caveat

A dry-run `cut` is only guaranteed not to touch the box if the arm got into
position through a dry-run move: `full_cut` with `dry_run`, or `move_to_center`
(top seam) / `move_to_seam` (side seams) with `dry_run`.

- A dry-run top-seam `cut` right after a real `move_to_center` slices sideways
  at the real 20 mm standoff. There is no insert, but a flap sticking up could
  still be hit.
- A dry-run side-seam `cut` right after a real `move_to_seam` slices at the
  real approach height. With `side_blade_insert_mm` at `0`, that is the same
  path a real cut takes: contact is possible.

The model doc will say both.

## Error handling

The gate raises `ValueError` for a missing frame, an out-of-bounds target, or a
failed transform. That is the error type the module already uses. `do_command`
passes it to the caller, and the busy guard is released as normal. The check
always runs before `motion.move`, so the arm is never sent a partial move. The
arm stays where the last completed move left it, the same as after `stop`.

## Testing

Added to `tests/test_control_dispatch.py`, using the fakes it already has
(`_RecordingMotion`, `_FakeRobotClient`):

1. Without `dry_run`, the gate adds nothing. `full_cut` makes only the two
   `transform_pose` calls `find_center` already makes, and the moves match
   what it sends today.
2. A dry-run top-seam `cut` sends no z inserts or retracts, and returns
   `skipped == ["top:insert", "top:retract", "top:insert", "top:retract"]`.
3. In a dry run with `side_blade_insert_mm: 16`, the close-seam retract is
   `-24`. The test needs a nonzero insert: at the default of `0`, the dry-run
   and real retracts are both `-40`, and the test would pass without the
   adjustment.
4. A dry-run `full_cut` returns
   `skipped == ["top:insert", "top:retract", "top:insert", "top:retract",
   "far:insert", "far:retract", "close:insert"]`.
5. A dry-run `move_to_center` descends to `standoff + clearance` above the box
   top.
6. A target outside the configured bounds raises before anything is sent to
   the motion client.
7. With no bounds configured, a dry run still runs and returns
   `bounds_checked: false`.
8. A frame the fake robot client cannot transform raises an error naming it,
   and nothing moves. `_FakeRobotClient` gains an option to raise for a named
   frame; today it never fails.
9. Config validation rejects only one of the two bounds being set, and a min
   axis greater than its max.

## Docs

The model doc gets a "Dry run" section: the new attributes, the caveat above,
and the recommended commissioning order. Run `full_cut` with `dry_run` first,
confirm every move's direction, then run it for real.
