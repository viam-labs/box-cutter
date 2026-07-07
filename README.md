# Module box-cutter

A Viam module that finds the center of a cardboard box in a depth camera frame and
plans/executes a cutting motion for a robot arm.

The `control` service segments the box by color, computes its center and the
tape-seam cut endpoints, transforms them into the robot's `world` frame, and can
move the cutting tool to the center or run a full cut sequence — all triggered via
`DoCommand`.

## Models

This module provides the following model(s):

- [`viam-labs:box-cutter:control`](viam-labs_box-cutter_control.md) — locate a box
  and plan/execute the cut. Supports the `find_center`, `move_to_center`, and
  `full_cut` commands.

## Development

Install dependencies and run the tests:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements-dev.txt
venv/bin/python -m pytest -v
```
