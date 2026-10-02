#!/usr/bin/env python3
"""Animate BB64 syndrome extraction as two mobile data grids around fixed ancillas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .animate_schedule import (
    DEFAULT_SCHEDULE,
    INDEX,
    Layer,
    Panel,
    derive_layers,
    element_name,
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _panel_by_half(layer: Layer, half: int) -> Panel:
    panels = [panel for panel in (layer.x_panel, layer.z_panel) if panel.half == half]
    if len(panels) != 1:
        raise ValueError(f"layer {layer.number} does not place data half {half} once")
    return panels[0]


def _cell(layer: Layer, check: int, *, bonded: bool) -> str:
    x_data = layer.x_panel.data_by_check[check]
    z_data = layer.z_panel.data_by_check[check]
    left = f"{layer.x_panel.half_name}{x_data:02d}"
    right = f"{layer.z_panel.half_name}{z_data:02d}"
    link = "--" if bonded else "  "
    return f"{left}{link}X{check:02d} Z{check:02d}{link}{right}"


def _render_grid(layer: Layer, *, bonded: bool) -> list[str]:
    lines = [
        "             x=0                 x=1                 x=2                 x=3"
    ]
    for yy in range(8):
        cells = "  ".join(
            _cell(layer, INDEX[(xx, yy)], bonded=bonded) for xx in range(4)
        )
        lines.append(f"y={yy} | {cells}")
    return lines


def _motion_line(layer: Layer, half: int) -> str:
    panel = _panel_by_half(layer, half)
    side = "left/X" if panel.kind == "X" else "right/Z"
    return (
        f"{panel.half_name} grid -> {side} contact side; "
        f"rigid move {element_name(panel.motion)} "
        f"(check->data delta {element_name(panel.delta)})"
    )


def render_position(layer: Layer, schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 82,
            f"BB64 SPATIAL VIEW | POSITION FOR LAYER {layer.number}/8 | bonds OFF",
            f"schedule {schedule_id}",
            "",
            _motion_line(layer, 0),
            _motion_line(layer, 1),
            "",
            "Each fixed supercell is Xcc Zcc; mobile data are shown on its two sides.",
            *_render_grid(layer, bonded=False),
        ]
    )


def render_contact(layer: Layer, schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 82,
            f"BB64 SPATIAL VIEW | CNOT LAYER {layer.number}/8 | 64 bonds ON",
            f"schedule {schedule_id}",
            "",
            _motion_line(layer, 0),
            _motion_line(layer, 1),
            "",
            "Bond directions: data--X means X ancilla -> data; Z--data means data -> Z ancilla.",
            *_render_grid(layer, bonded=True),
        ]
    )


def render_prepare(schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 82,
            "BB64 SPATIAL VIEW | PREPARE",
            f"schedule {schedule_id}",
            "",
            "The fixed 4 x 8 supergrid contains X00..X31 and Z00..Z31.",
            "Reset every X ancilla to |+> and every Z ancilla to |0>.",
            "The parking geometry of mobile grids L00..L31 and R00..R31 is unspecified.",
            "No CNOT bonds are active.",
        ]
    )


def render_measure(schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 82,
            "BB64 SPATIAL VIEW | MEASURE",
            f"schedule {schedule_id}",
            "",
            "Measure fixed X00..X31 in the X basis and fixed Z00..Z31 in the Z basis.",
            "The mobile data grids remain encoded; their post-round parking move is unspecified.",
            "No CNOT bonds are active.",
        ]
    )


def animation_frames(layers: Iterable[Layer], schedule_id: str) -> list[str]:
    frames = [render_prepare(schedule_id)]
    for layer in layers:
        frames.extend((render_position(layer, schedule_id), render_contact(layer, schedule_id)))
    frames.append(render_measure(schedule_id))
    return frames


def render_all(layers: Iterable[Layer], schedule_id: str) -> str:
    preamble = "\n".join(
        [
            "BB64 NEUTRAL-ATOM SPATIAL CONTACT ANIMATION",
            "",
            "At each of 32 fixed sites, Xcc Zcc are stationary syndrome atoms.",
            "The L and R 32-atom data grids move as rigid group translates to the",
            "left/X or right/Z side. A -- link appears only during an active CNOT.",
            "Data identities visibly move between frames; each contact frame contains",
            "all 64 data atoms, all 64 syndrome atoms, and all 64 scheduled CNOTs.",
            "",
            "Coordinates use x=0..3, y=0..7 with (x+4,y)=(x,y+4). The frames",
            "show exact contact configurations, not continuous collision-free paths.",
            "",
        ]
    )
    return preamble + "\n\n".join(animation_frames(layers, schedule_id)) + "\n"


def animate(frames: Iterable[str], fps: float, cycles: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    delay = 1.0 / fps
    try:
        for _ in range(cycles):
            for frame in frames:
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write(frame)
                sys.stdout.write("\n\nCtrl-C stops the animation.\n")
                sys.stdout.flush()
                time.sleep(delay)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--animate", action="store_true", help="animate with ANSI screen clears")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--layer", type=int, choices=range(1, 9))
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--output", type=Path, help="save all static position/contact frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = _load(args.schedule)
    layers = derive_layers(schedule)
    schedule_id = str(schedule.get("schedule_id", "unknown"))
    rendered = render_all(layers, schedule_id)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="ascii")
        print(f"saved {args.output}", file=sys.stderr)
    if args.layer is not None:
        layer = layers[args.layer - 1]
        renderer = render_position if args.position_only else render_contact
        print(renderer(layer, schedule_id))
    elif args.animate:
        animate(animation_frames(layers, schedule_id), args.fps, args.cycles)
    elif args.output is None:
        print(rendered, end="")


if __name__ == "__main__":
    main()
