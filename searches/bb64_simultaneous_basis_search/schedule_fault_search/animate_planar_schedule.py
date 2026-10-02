#!/usr/bin/env python3
"""Render the BB64 schedule on a finite planar C8 x C4 tweezer array."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .animate_schedule import (
    DEFAULT_SCHEDULE,
    ELEMENTS,
    Layer,
    Panel,
    derive_layers,
)


Planar = tuple[int, int]
PLANAR_ELEMENTS: tuple[Planar, ...] = tuple(
    (uu, vv) for uu in range(8) for vv in range(4)
)
PLANAR_INDEX: dict[Planar, int] = {
    ((xx + yy) % 8, xx): index for index, (xx, yy) in enumerate(ELEMENTS)
}


@dataclass(frozen=True)
class Placement:
    half: int
    kind: str
    motion: Planar

    @property
    def half_name(self) -> str:
        return "L" if self.half == 0 else "R"

    @property
    def side(self) -> str:
        return "left/X" if self.kind == "X" else "right/Z"


@dataclass(frozen=True)
class Transition:
    previous: Placement
    target: Placement
    step: Planar
    wrapped: frozenset[Planar]

    @property
    def changes_side(self) -> bool:
        return self.previous.kind != self.target.kind

    @property
    def corridor(self) -> str:
        if not self.changes_side:
            return "same contact side"
        return "upper exterior corridor" if self.target.half == 0 else "lower exterior corridor"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def to_planar(element: tuple[int, int]) -> Planar:
    """Map the saved twisted chart to ordinary C8 x C4 coordinates."""
    xx, yy = element
    return (xx + yy) % 8, xx % 4


def planar_add(left: Planar, right: Planar) -> Planar:
    return (left[0] + right[0]) % 8, (left[1] + right[1]) % 4


def _signed_component(previous: int, target: int, period: int) -> int:
    value = (target - previous) % period
    if value > period // 2:
        value -= period
    return value


def signed_step(previous: Planar, target: Planar) -> Planar:
    return (
        _signed_component(previous[0], target[0], 8),
        _signed_component(previous[1], target[1], 4),
    )


def vector_name(vector: Planar) -> str:
    return f"({vector[0]:+d},{vector[1]:+d})"


def placement(panel: Panel) -> Placement:
    return Placement(half=panel.half, kind=panel.kind, motion=to_planar(panel.motion))


def placements(layer: Layer) -> dict[int, Placement]:
    result = {
        layer.x_panel.half: placement(layer.x_panel),
        layer.z_panel.half: placement(layer.z_panel),
    }
    if set(result) != {0, 1}:
        raise ValueError(f"layer {layer.number} does not place both data grids")
    return result


def initial_placements() -> dict[int, Placement]:
    # A declared planar parking convention, not information from the code.
    return {
        0: Placement(half=0, kind="X", motion=(0, 0)),
        1: Placement(half=1, kind="Z", motion=(0, 0)),
    }


def transition(previous: Placement, target: Placement) -> Transition:
    step = signed_step(previous.motion, target.motion)
    wrapped: set[Planar] = set()
    for base in PLANAR_ELEMENTS:
        start = planar_add(base, previous.motion)
        raw_target = start[0] + step[0], start[1] + step[1]
        if not (0 <= raw_target[0] < 8 and 0 <= raw_target[1] < 4):
            wrapped.add(base)
        if planar_add(start, step) != planar_add(base, target.motion):
            raise ValueError("signed planar step does not reach its target placement")
    return Transition(
        previous=previous,
        target=target,
        step=step,
        wrapped=frozenset(wrapped),
    )


def transitions(previous: dict[int, Placement], layer: Layer) -> dict[int, Transition]:
    target = placements(layer)
    return {half: transition(previous[half], target[half]) for half in (0, 1)}


def data_coordinate(panel: Panel, check: int) -> Planar:
    return to_planar(ELEMENTS[panel.data_by_check[check]])


def _cell(
    layer: Layer,
    check_coordinate: Planar,
    layer_transitions: dict[int, Transition],
    *,
    bonded: bool,
) -> str:
    check = PLANAR_INDEX[check_coordinate]
    x_data = data_coordinate(layer.x_panel, check)
    z_data = data_coordinate(layer.z_panel, check)
    x_wrap = x_data in layer_transitions[layer.x_panel.half].wrapped
    z_wrap = z_data in layer_transitions[layer.z_panel.half].wrapped
    left = f"{'*' if x_wrap else ' '}{layer.x_panel.half_name}{x_data[0]}{x_data[1]}"
    right = f"{'*' if z_wrap else ' '}{layer.z_panel.half_name}{z_data[0]}{z_data[1]}"
    link = "--" if bonded else "  "
    return f"{left}{link}XZ{check_coordinate[0]}{check_coordinate[1]}{link}{right}"


def _render_grid(
    layer: Layer,
    layer_transitions: dict[int, Transition],
    *,
    bonded: bool,
) -> list[str]:
    lines = [
        ("       " + " ".join(f"u={uu}".center(16) for uu in range(8))).rstrip()
    ]
    for vv in range(4):
        cells = " ".join(
            _cell(layer, (uu, vv), layer_transitions, bonded=bonded)
            for uu in range(8)
        )
        lines.append(f"v={vv} | {cells}")
    return lines


def _wrap_names(change: Transition) -> str:
    prefix = change.target.half_name
    names = " ".join(f"{prefix}{uu}{vv}" for uu, vv in sorted(change.wrapped))
    return names if names else "none"


def _routing_lines(change: Transition) -> list[str]:
    return [
        (
            f"{change.target.half_name}: {change.previous.side} -> {change.target.side}; "
            f"planar step {vector_name(change.step)}; {change.corridor}; "
            f"wrap atoms={len(change.wrapped)}"
        ),
        f"    perimeter-bypass set: {_wrap_names(change)}",
    ]


def render_position(
    layer: Layer,
    layer_transitions: dict[int, Transition],
    schedule_id: str,
) -> str:
    return "\n".join(
        [
            "=" * 141,
            f"BB64 STRICT 2D PLANE | MOVE TO LAYER {layer.number}/8 | bonds OFF",
            f"schedule {schedule_id}",
            "",
            *_routing_lines(layer_transitions[0]),
            *_routing_lines(layer_transitions[1]),
            "",
            "* marks a data atom routed through an exterior lane at a finite-array boundary.",
            *_render_grid(layer, layer_transitions, bonded=False),
        ]
    )


def render_contact(
    layer: Layer,
    layer_transitions: dict[int, Transition],
    schedule_id: str,
) -> str:
    return "\n".join(
        [
            "=" * 141,
            f"BB64 STRICT 2D PLANE | CNOT LAYER {layer.number}/8 | 64 bonds ON",
            f"schedule {schedule_id}",
            "",
            *_routing_lines(layer_transitions[0]),
            *_routing_lines(layer_transitions[1]),
            "",
            "In A--XZuv--B, X controls A and B controls Z. XZuv remains fixed.",
            *_render_grid(layer, layer_transitions, bonded=True),
        ]
    )


def render_prepare(schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 141,
            "BB64 STRICT 2D PLANE | PREPARE",
            f"schedule {schedule_id}",
            "",
            "Fixed 8 x 4 syndrome supergrid: XZuv contains one X and one Z ancilla.",
            "Reset X ancillas to |+> and Z ancillas to |0>.",
            "Declared initial parking: L at left/X shift (+0,+0), R at right/Z shift (+0,+0).",
            "No CNOT bonds are active.",
        ]
    )


def render_measure(schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 141,
            "BB64 STRICT 2D PLANE | MEASURE",
            f"schedule {schedule_id}",
            "",
            "Measure all fixed X ancillas in X and all fixed Z ancillas in Z.",
            "The final return-to-parking move is not part of this syndrome round.",
            "No CNOT bonds are active.",
        ]
    )


def animation_frames(layers: Iterable[Layer], schedule_id: str) -> list[str]:
    previous = initial_placements()
    frames = [render_prepare(schedule_id)]
    for layer in layers:
        changes = transitions(previous, layer)
        frames.extend(
            (
                render_position(layer, changes, schedule_id),
                render_contact(layer, changes, schedule_id),
            )
        )
        previous = placements(layer)
    frames.append(render_measure(schedule_id))
    return frames


def render_all(layers: Iterable[Layer], schedule_id: str) -> str:
    preamble = "\n".join(
        [
            "BB64 STRICT-2D OPTICAL-TWEEZER SCHEDULE",
            "",
            "The code coordinates are relabeled as an ordinary finite C8 x C4 rectangle:",
            "u=0..7 horizontally and v=0..3 vertically. There is no edge identification",
            "in the drawing. XZuv are 64 stationary syndrome atoms in 32 fixed pairs;",
            "Luv and Ruv are the two mobile 32-atom data grids.",
            "",
            "A finite plane cannot realize a cyclic wrap as one Euclidean rigid motion.",
            "Non-wrapping atoms share the listed displacement; boundary atoms marked *",
            "must use exterior bypass lanes. L side exchanges use the upper corridor and",
            "R side exchanges use the lower corridor to keep the two mobile grids apart.",
            "These frames specify exact endpoints and routing classes, not calibrated or",
            "formally collision-certified continuous tweezer trajectories.",
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
    parser.add_argument("--move-only", action="store_true")
    parser.add_argument("--output", type=Path, help="save all static move/contact frames")
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
        index = args.layer - 1
        previous = initial_placements()
        for earlier in layers[:index]:
            previous = placements(earlier)
        changes = transitions(previous, layers[index])
        renderer = render_position if args.move_only else render_contact
        print(renderer(layers[index], changes, schedule_id))
    elif args.animate:
        animate(animation_frames(layers, schedule_id), args.fps, args.cycles)
    elif args.output is None:
        print(rendered, end="")


if __name__ == "__main__":
    main()
