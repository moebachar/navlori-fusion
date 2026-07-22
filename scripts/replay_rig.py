"""Shared definition of the kinematic REPLAY_RIG used by every replay world.

One source of truth imported by both `build_iln20_webots_world.py` (emits the
rig at build time) and `run_replay.py` (patches/upgrades the rig in an
existing world).

Design constraints (learned the hard way, 2026-07-21):
  * NO Physics node, NO boundingObject anywhere in the rig. A physics-less
    Robot is moved purely by supervisor field writes -- the constraint solver
    never touches it, so it cannot explode, drift, or shed parts (all of
    which the articulated Tiago++ PROTO did when pose-anchored).
  * The camera is EXACTLY the proven iteration-1 sensor: 640x480,
    fieldOfView 1.0472, at (0.08, 0, 1.19) looking along +x (Webots FLU),
    name "head_front_camera" -- identical to TIAGO's head camera, so the
    dataset is bit-identical to what a real TIAGO would record.
  * Every body shape stays BEHIND the lens plane (max +x face <= 0.075 m for
    parts near camera height) or far outside the view frustum -- the robot
    must never see itself.
  * The body is a TIAGO++ silhouette (dark PMB-2 base + orange trim, white
    torso, dual tucked arms, pan-tilt head with dark visor) so replays LOOK
    like a TIAGO driving, while keeping the kinematic guarantees.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

# PAL TIAGO palette
_WHITE = "0.93 0.93 0.94"
_GREY = "0.45 0.45 0.48"
_DARK = "0.16 0.16 0.18"
_WHEEL = "0.10 0.10 0.11"
_ORANGE = "0.93 0.40 0.05"
_VISOR = "0.07 0.07 0.09"

RIG_TEMPLATE = """DEF REPLAY_RIG Robot {{
  translation {x:.4f} {y:.4f} 0
  rotation 0 0 1 {yaw:.5f}
  name "replay_rig"
  controller "replay_collector"
  supervisor TRUE
  children [
    Pose {{
      translation 0 0.2022 0.0985
      rotation 1 0 0 1.5708
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(wheel)s roughness 0.8 metalness 0.1 }}
          geometry Cylinder {{ radius 0.0985 height 0.045 }}
        }}
      ]
    }}
    Pose {{
      translation 0 -0.2022 0.0985
      rotation 1 0 0 1.5708
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(wheel)s roughness 0.8 metalness 0.1 }}
          geometry Cylinder {{ radius 0.0985 height 0.045 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 0.19
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(dark)s roughness 0.6 metalness 0.3 }}
          geometry Cylinder {{ radius 0.27 height 0.30 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 0.315
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(orange)s roughness 0.5 metalness 0.1 }}
          geometry Cylinder {{ radius 0.272 height 0.035 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 0.62
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(white)s roughness 0.55 metalness 0.05 }}
          geometry Box {{ size 0.28 0.32 0.56 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 0.97
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(white)s roughness 0.55 metalness 0.05 }}
          geometry Box {{ size 0.22 0.50 0.14 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0.29 0.95
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(orange)s roughness 0.5 metalness 0.1 }}
          geometry Sphere {{ radius 0.062 }}
        }}
      ]
    }}
    Pose {{
      translation 0 -0.29 0.95
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(orange)s roughness 0.5 metalness 0.1 }}
          geometry Sphere {{ radius 0.062 }}
        }}
      ]
    }}
    Pose {{
      translation 0.05 0.305 0.815
      rotation 0 1 0 0.30
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(grey)s roughness 0.6 metalness 0.2 }}
          geometry Capsule {{ radius 0.048 height 0.26 }}
        }}
      ]
    }}
    Pose {{
      translation 0.05 -0.305 0.815
      rotation 0 1 0 0.30
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(grey)s roughness 0.6 metalness 0.2 }}
          geometry Capsule {{ radius 0.048 height 0.26 }}
        }}
      ]
    }}
    Pose {{
      translation 0.16 0.26 0.66
      rotation 0 1 0 1.35
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(grey)s roughness 0.6 metalness 0.2 }}
          geometry Capsule {{ radius 0.042 height 0.24 }}
        }}
      ]
    }}
    Pose {{
      translation 0.16 -0.26 0.66
      rotation 0 1 0 1.35
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(grey)s roughness 0.6 metalness 0.2 }}
          geometry Capsule {{ radius 0.042 height 0.24 }}
        }}
      ]
    }}
    Pose {{
      translation 0.27 0.25 0.63
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(dark)s roughness 0.7 metalness 0.2 }}
          geometry Sphere {{ radius 0.045 }}
        }}
      ]
    }}
    Pose {{
      translation 0.27 -0.25 0.63
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(dark)s roughness 0.7 metalness 0.2 }}
          geometry Sphere {{ radius 0.045 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 1.06
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(grey)s roughness 0.6 metalness 0.2 }}
          geometry Cylinder {{ radius 0.05 height 0.06 }}
        }}
      ]
    }}
    Pose {{
      translation -0.025 0 1.16
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(white)s roughness 0.5 metalness 0.05 }}
          geometry Box {{ size 0.20 0.24 0.155 }}
        }}
      ]
    }}
    Pose {{
      translation 0.065 0 1.175
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor %(visor)s roughness 0.3 metalness 0.4 }}
          geometry Box {{ size 0.02 0.20 0.06 }}
        }}
      ]
    }}
    Camera {{
      translation 0.08 0 1.19
      name "head_front_camera"
      fieldOfView 1.0472
      width 640
      height 480
      far 50
    }}
  ]
}}
""" % {"white": _WHITE, "grey": _GREY, "dark": _DARK, "wheel": _WHEEL,
       "orange": _ORANGE, "visor": _VISOR}


_TIAGO_HEAD_RE = re.compile(r"DEF TIAGO Tiago\+\+ \{")
_RIG_HEAD_RE = re.compile(r"DEF REPLAY_RIG Robot \{")


def _block_end(text: str, brace_open_idx: int) -> int:
    depth = 0
    for i in range(brace_open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def format_rig(x: float, y: float, yaw: float) -> str:
    return RIG_TEMPLATE.format(x=x, y=y, yaw=yaw)


def ensure_replay_rig(world: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Make `world` contain the CURRENT replay rig.

    - If a Tiago++ node is present, replace it (keeping translation/rotation)
      and drop its EXTERNPROTO download.
    - If an (older) REPLAY_RIG is present, replace it with the current
      template (keeping translation/rotation) -- worlds staged before a rig
      redesign get upgraded automatically.
    Backup: <world>.bak-tiago (only written when the file changes).
    """
    text = world.read_text(encoding="utf-8")

    m = _RIG_HEAD_RE.search(text) or _TIAGO_HEAD_RE.search(text)
    if not m:
        return False, "neither REPLAY_RIG nor DEF TIAGO Tiago++ found"
    was_tiago = m.re is _TIAGO_HEAD_RE

    end = _block_end(text, m.end() - 1)
    block = text[m.start():end]
    mt = re.search(r"translation\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", block)
    mr = re.search(r"rotation\s+0\s+0\s+1\s+(-?[\d.]+)", block)
    if not mt:
        return False, "robot block has no parsable translation"
    rig = format_rig(float(mt.group(1)), float(mt.group(2)),
                     float(mr.group(1)) if mr else 0.0)
    # exact block-for-block replacement (both end at the closing brace) --
    # keeps the surrounding whitespace untouched so a second call is a no-op
    new_text = text[:m.start()] + rig.rstrip("\n") + text[end:]
    # the Tiago++ PROTO is no longer used -- drop its EXTERNPROTO download
    new_text = re.sub(r'EXTERNPROTO "[^"]*Tiago\+\+\.proto"\n', "", new_text)

    if new_text == text:
        return True, "REPLAY_RIG already current"
    if not dry_run:
        bak = world.with_suffix(world.suffix + ".bak-tiago")
        shutil.copy2(world, bak)
        world.write_text(new_text, encoding="utf-8")
    what = "replaced Tiago++ with" if was_tiago else "upgraded existing rig to"
    return True, f"{what} current TIAGO-shell REPLAY_RIG (backup .bak-tiago)"
