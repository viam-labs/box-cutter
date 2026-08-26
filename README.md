# Module box-cutter

A Viam module that finds a cardboard box in a depth camera frame and cuts its
three tape seams with a blade on a robot arm.

The `control` service segments the box by color, derives its geometry in the
robot's `world` frame, then works seam by seam: it visually servos the blade onto
each seam before slicing it. Everything is triggered via `DoCommand`.

## Models

This module provides the following model(s):

- [`viam-labs:box-cutter:control`](viam-labs_box-cutter_control.md) — locate a box
  and cut its top, far, and close seams. Supports the `set_box`, `home`,
  `find_center`, `move_to_center`, `converge`, `cut`, and `full_cut` commands.

## Development

Install dependencies and run the tests:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements-dev.txt
venv/bin/python -m pytest -v
```
