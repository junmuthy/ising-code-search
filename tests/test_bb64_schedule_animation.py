from __future__ import annotations

import json
from pathlib import Path

from bb64_simultaneous_basis_search.schedule_fault_search.animate_schedule import (
    add,
    derive_layers,
    inverse,
    render_all,
    render_frame,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = (
    ROOT
    / "bb64_simultaneous_basis_search"
    / "schedule_fault_search"
    / "baseline_schedule.json"
)
ANIMATION = SCHEDULE.with_name("schedule_animation.txt")


def load_schedule() -> dict:
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def test_twisted_group_inverses() -> None:
    assert inverse((0, 0)) == (0, 0)
    assert inverse((1, 0)) == (3, 4)
    assert inverse((0, 1)) == (0, 7)
    assert add((3, 4), (1, 0)) == (0, 0)


def test_every_layer_is_two_rigid_translations() -> None:
    layers = derive_layers(load_schedule())
    expected = (
        ("L", (0, 0), "R", (0, 1)),
        ("R", (3, 4), "L", (0, 7)),
        ("L", (0, 1), "R", (0, 7)),
        ("L", (1, 0), "R", (0, 0)),
        ("R", (0, 0), "L", (1, 0)),
        ("R", (0, 7), "L", (0, 1)),
        ("L", (0, 7), "R", (3, 4)),
        ("R", (0, 1), "L", (0, 0)),
    )
    observed = tuple(
        (
            layer.x_panel.half_name,
            layer.x_panel.delta,
            layer.z_panel.half_name,
            layer.z_panel.delta,
        )
        for layer in layers
    )
    assert observed == expected


def test_rendered_frames_are_ascii_and_cover_all_atoms() -> None:
    schedule = load_schedule()
    layers = derive_layers(schedule)
    rendered = render_all(layers, schedule["schedule_id"])
    rendered.encode("ascii")
    assert rendered.count("64 parallel CNOTs") == 8
    assert "X00->L00" in render_frame(layers[0], schedule["schedule_id"])
    assert "R01->Z00" in render_frame(layers[0], schedule["schedule_id"])
    for layer in layers:
        assert sorted(layer.x_panel.data_by_check + layer.z_panel.data_by_check) == (
            [value for value in range(32) for _ in range(2)]
        )


def test_saved_animation_matches_canonical_schedule() -> None:
    schedule = load_schedule()
    rendered = render_all(derive_layers(schedule), schedule["schedule_id"])
    assert ANIMATION.read_text(encoding="ascii") == rendered
