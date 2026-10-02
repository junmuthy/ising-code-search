#!/usr/bin/env python3
"""Render the canonical BB64 syndrome schedule as an ASCII atom animation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Element = tuple[int, int]
ELEMENTS: tuple[Element, ...] = tuple(
    (xx, yy) for xx in range(4) for yy in range(8)
)
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
DEFAULT_SCHEDULE = Path(__file__).resolve().parent / "baseline_schedule.json"


@dataclass(frozen=True)
class Panel:
    kind: str
    half: int
    delta: Element
    data_by_check: tuple[int, ...]

    @property
    def half_name(self) -> str:
        return "L" if self.half == 0 else "R"

    @property
    def motion(self) -> Element:
        """Rigid motion that brings data at check + delta onto the check."""
        return inverse(self.delta)


@dataclass(frozen=True)
class Layer:
    number: int
    x_panel: Panel
    z_panel: Panel


def add(left: Element, right: Element) -> Element:
    """Add coordinates on the saved twisted C8 x C4 presentation."""
    raw_x = left[0] + right[0]
    return raw_x % 4, (left[1] + right[1] + 4 * (raw_x // 4)) % 8


def inverse(element: Element) -> Element:
    for candidate in ELEMENTS:
        if add(element, candidate) == (0, 0):
            return candidate
    raise ValueError(f"element has no inverse: {element}")


def difference(left: Element, right: Element) -> Element:
    """Return delta such that left + delta equals right."""
    return add(inverse(left), right)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _derive_panel(gates: Iterable[dict[str, Any]], kind: str) -> Panel:
    selected = [gate for gate in gates if gate["type"] == kind]
    if len(selected) != 32:
        raise ValueError(f"expected 32 {kind} gates, found {len(selected)}")
    checks = {int(gate["check"]) for gate in selected}
    if checks != set(range(32)):
        raise ValueError(f"{kind} panel does not use every check exactly once")
    halves = {int(gate["data"]) // 32 for gate in selected}
    if len(halves) != 1:
        raise ValueError(f"{kind} panel is not matched to one rigid data half")
    half = halves.pop()
    data_by_check = [-1] * 32
    deltas: set[Element] = set()
    for gate in selected:
        check = int(gate["check"])
        data = int(gate["data"])
        data_within = data % 32
        data_by_check[check] = data_within
        deltas.add(difference(ELEMENTS[check], ELEMENTS[data_within]))
    if len(deltas) != 1:
        raise ValueError(f"{kind} panel is not a uniform group translation: {deltas}")
    return Panel(
        kind=kind,
        half=half,
        delta=deltas.pop(),
        data_by_check=tuple(data_by_check),
    )


def derive_layers(schedule: dict[str, Any]) -> tuple[Layer, ...]:
    raw_layers = schedule.get("layers", [])
    if len(raw_layers) != 8:
        raise ValueError(f"expected eight layers, found {len(raw_layers)}")
    layers = tuple(
        Layer(
            number=int(raw["round"]),
            x_panel=_derive_panel(raw["gates"], "X"),
            z_panel=_derive_panel(raw["gates"], "Z"),
        )
        for raw in raw_layers
    )
    expected_numbers = tuple(range(1, 9))
    if tuple(layer.number for layer in layers) != expected_numbers:
        raise ValueError("layers are not numbered consecutively from one through eight")
    for layer, raw in zip(layers, raw_layers, strict=True):
        data = [int(gate["data"]) for gate in raw["gates"]]
        ancillas = [(str(gate["type"]), int(gate["check"])) for gate in raw["gates"]]
        if sorted(data) != list(range(64)):
            raise ValueError(f"layer {layer.number} does not use every data atom once")
        if len(set(ancillas)) != 64:
            raise ValueError(f"layer {layer.number} does not use every syndrome atom once")
    return layers


def element_name(element: Element) -> str:
    names = {
        (0, 0): "0",
        (1, 0): "+x",
        (3, 4): "-x",
        (0, 1): "+y",
        (0, 7): "-y",
    }
    return names.get(element, f"({element[0]},{element[1]})")


def _cell(panel: Panel, check: int) -> str:
    data = panel.data_by_check[check]
    if panel.kind == "X":
        # X-check ancilla is the CNOT control.
        return f"X{check:02d}->{panel.half_name}{data:02d}"
    # Z-check data is the CNOT control.
    return f"{panel.half_name}{data:02d}->Z{check:02d}"


def _render_panel(panel: Panel) -> list[str]:
    direction = "ancilla -> data" if panel.kind == "X" else "data -> ancilla"
    lines = [
        (
            f"{panel.kind} syndrome plane (fixed): {panel.half_name} data grid; "
            f"check->data delta={element_name(panel.delta)}; "
            f"move data by {element_name(panel.motion)}; {direction}"
        ),
        "       y=0     y=1     y=2     y=3     y=4     y=5     y=6     y=7",
    ]
    for xx in range(4):
        cells = " ".join(_cell(panel, INDEX[(xx, yy)]) for yy in range(8))
        lines.append(f"x={xx} | {cells}")
    return lines


def render_frame(layer: Layer, schedule_id: str) -> str:
    lines = [
        "=" * 78,
        f"BB64 syndrome extraction | layer {layer.number}/8 | 64 parallel CNOTs",
        f"schedule {schedule_id}",
        "",
        *_render_panel(layer.x_panel),
        "",
        *_render_panel(layer.z_panel),
        "",
        "Legend: Xcc->Ddd is X-ancilla cc controlling data atom Ddd.",
        "        Ddd->Zcc is data atom Ddd controlling Z-ancilla cc.",
        "        Lnn has global data ID nn; Rnn has global data ID 32+nn.",
    ]
    return "\n".join(lines)


def render_prepare(schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 78,
            "BB64 syndrome extraction | PREPARE",
            f"schedule {schedule_id}",
            "",
            "Fixed syndrome atoms X00..X31: reset to |+> (future CNOT controls).",
            "Fixed syndrome atoms Z00..Z31: reset to |0> (future CNOT targets).",
            "Data atoms L00..L31 and R00..R31 retain the encoded state.",
            "No CNOT bonds are active in this frame.",
        ]
    )


def render_measure(schedule_id: str) -> str:
    return "\n".join(
        [
            "=" * 78,
            "BB64 syndrome extraction | MEASURE",
            f"schedule {schedule_id}",
            "",
            "Measure fixed syndrome atoms X00..X31 in the X basis.",
            "Measure fixed syndrome atoms Z00..Z31 in the Z basis.",
            "Data atoms L00..L31 and R00..R31 remain encoded.",
            "No CNOT bonds are active in this frame.",
        ]
    )


def render_all(layers: Iterable[Layer], schedule_id: str) -> str:
    preamble = [
        "BB64 IDEALIZED RIGID-TRANSLATION SYNDROME ANIMATION",
        "",
        "The 32 X and 32 Z syndrome atoms stay fixed in two displayed planes.",
        "At each layer, the L and R 32-atom data grids are routed onto those planes",
        "with the stated rigid translation; every displayed -> is an active CNOT.",
        "Coordinates use x=0..3, y=0..7 and the twisted boundary",
        "(x+4,y) = (x,y+4). This is a combinatorial routing view, not a",
        "continuous collision-free tweezer trajectory.",
        "",
    ]
    frames = [
        render_prepare(schedule_id),
        *(render_frame(layer, schedule_id) for layer in layers),
        render_measure(schedule_id),
    ]
    return "\n".join(preamble) + "\n\n".join(frames) + "\n"


def animate(frames: list[str], fps: float, cycles: int) -> None:
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
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--frame", type=int, choices=range(1, 9))
    parser.add_argument("--output", type=Path, help="save all eight static ASCII frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = _load(args.schedule)
    layers = derive_layers(schedule)
    schedule_id = str(schedule.get("schedule_id", "unknown"))
    all_frames = render_all(layers, schedule_id)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(all_frames, encoding="ascii")
        print(f"saved {args.output}", file=sys.stderr)
    if args.frame is not None:
        print(render_frame(layers[args.frame - 1], schedule_id))
    elif args.animate:
        animate(
            [
                render_prepare(schedule_id),
                *(render_frame(layer, schedule_id) for layer in layers),
                render_measure(schedule_id),
            ],
            args.fps,
            args.cycles,
        )
    elif args.output is None:
        print(all_frames, end="")


if __name__ == "__main__":
    main()
