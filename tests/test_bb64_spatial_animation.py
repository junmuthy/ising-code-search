from __future__ import annotations

import json
from pathlib import Path

from bb64_simultaneous_basis_search.schedule_fault_search.animate_schedule import (
    derive_layers,
)
from bb64_simultaneous_basis_search.schedule_fault_search.animate_spatial_schedule import (
    animation_frames,
    render_all,
    render_contact,
    render_position,
)


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "bb64_simultaneous_basis_search" / "schedule_fault_search"
SCHEDULE = DIRECTORY / "baseline_schedule.json"
ANIMATION = DIRECTORY / "spatial_schedule_animation.txt"


def load_schedule() -> dict:
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def test_spatial_contact_frames_show_every_atom_and_bond() -> None:
    schedule = load_schedule()
    for layer in derive_layers(schedule):
        position = render_position(layer, schedule["schedule_id"])
        contact = render_contact(layer, schedule["schedule_id"])
        position_grid = "\n".join(
            line for line in position.splitlines() if line.startswith("y=")
        )
        contact_grid = "\n".join(
            line for line in contact.splitlines() if line.startswith("y=")
        )
        assert "--" not in position_grid
        assert contact_grid.count("--") == 64
        for check in range(32):
            assert contact.count(f"X{check:02d}") == 1
            assert contact.count(f"Z{check:02d}") == 1
        for data in range(32):
            assert contact.count(f"L{data:02d}") == 1
            assert contact.count(f"R{data:02d}") == 1


def test_spatial_view_exposes_side_swaps_and_translations() -> None:
    schedule = load_schedule()
    layers = derive_layers(schedule)
    first = render_contact(layers[0], schedule["schedule_id"])
    second = render_contact(layers[1], schedule["schedule_id"])
    assert "L00--X00 Z00--R01" in first
    assert "R28--X00 Z00--L07" in second
    assert "L grid -> left/X contact side; rigid move 0" in first
    assert "L grid -> right/Z contact side; rigid move +y" in second


def test_complete_spatial_animation_is_ascii_and_saved_exactly() -> None:
    schedule = load_schedule()
    layers = derive_layers(schedule)
    rendered = render_all(layers, schedule["schedule_id"])
    rendered.encode("ascii")
    frames = animation_frames(layers, schedule["schedule_id"])
    assert len(frames) == 18
    assert rendered.count("64 bonds ON") == 8
    assert ANIMATION.read_text(encoding="ascii") == rendered
