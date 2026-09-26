# Calibrating the box cutter

How to calibrate the box bot: an arm with a camera and a cutting tool. All
commands below are sent as `DoCommand` payloads, for example from the Viam
app's control tab.

## Step 1: home position and box depth

Choose a home position (`home_xyz`) and park the arm there with
`{"command": "home"}`. Physically measure the distance from the camera lens to
the top of the box. This is `depth_mm`, the first box measurement. It sets how
far the arm descends when it moves to the center of the box.

## Step 2: box center pixel

In the Viam app's camera control tab, turn on mouse coordinates and hover over
the center of the box. The first value is `u` and the second is `v`.

## Step 3: box size

Measure two more values on the box:

- `flap_width_mm`: the width of a flap, from the center seam to the side of the
  box.
- `box_height_mm`: the length of the top seam.

## Step 4: apply the box

Send all five values with `set_box`:

```json
{ "command": "set_box", "depth_mm": 525, "u": 358, "v": 225,
  "flap_width_mm": 120, "box_height_mm": 280 }
```

Or apply a saved preset: `{"command": "set_box", "preset": "box_1"}`.

Presets are hard-coded in `BOX_PRESETS` in `src/models/control.py`, not in the
service config. Saving a new preset means editing that file and redeploying the
module. `box_2` to `box_5` have no measured `box_height_mm` yet (it is `-1`),
so `set_box` rejects them until one is added.

## Step 5: cell geometry

Set these config attributes on the service. They're needed to stage the side
seams at the right place and height.

- `stopper_x_mm`: the world X of the box's close edge, which rests against the
  stopper. To measure it, `home`, then `jog` the knife tip until it sits over
  the close edge. `jog` moves the tool in its own frame, up to 50 mm per axis
  per call, for example `{"command": "jog", "x": 10}`. Its response includes
  `world_after`: use that pose's `x`. Its `world_delta` shows which world axis
  each tool axis moved, so check that first.
- `knife_tip_to_table_mm`: the height of the knife tip above the table at home.
- `base_plate_height_mm`: the thickness of the plate the arm is bolted to.

## Step 6: blade column

`converge` lines the seam up with the blade's pixel column, `blade_x_px`. If
`converge` reports success but the blade ends up beside the seam rather than
over it, adjust `blade_x_px` and try again.

## Step 7: cutting depth

This is the most experimental part, because it works with the blade. If the
blade goes in too far or not far enough, first recheck `depth_mm` and
`stopper_x_mm`. Then adjust the insert depths:

- `top_blade_insert_mm` for the top seam.
- `side_blade_insert_mm` for the far and close seams. The close seam currently
  inserts a hard-coded 7 mm deeper than this (in `_cut_side_seam`), because
  the blade sits further from the tape there.

## Running the module

```json
{ "command": "home" }
{ "command": "set_box", "preset": "box_1" }
{ "command": "move_to_center" }
{ "command": "converge", "seam": "top" }
{ "command": "cut", "seam": "top" }
{ "command": "move_to_seam", "seam": "far" }
{ "command": "converge", "seam": "far" }
{ "command": "cut", "seam": "far" }
{ "command": "move_to_seam", "seam": "close" }
{ "command": "converge", "seam": "close" }
{ "command": "cut", "seam": "close" }
{ "command": "home" }
```

- After each `converge`, check that the blade is over the seam before you cut.
- Pass `seam` to `converge` on the side seams. Without it, `converge` servos
  with the top-seam gain.
- `cut` can leave out `seam` and work it out from the tool's position. Passing
  it is safer.

Once each step works, `{"command": "full_cut"}` runs the whole sequence, from
`home` back to `home`. Run `set_box` first: `full_cut` doesn't set the box.
