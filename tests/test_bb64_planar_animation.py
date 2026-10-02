from __future__ import annotations

import json
from pathlib import Path

from searches.bb64_simultaneous_basis_search.schedule_fault_search.animate_planar_schedule import (
    animation_frames,
    initial_placements,
    placements,
    planar_add,
    render_all,
    render_contact,
    render_position,
    to_planar,
    transitions,
)
from searches.bb64_simultaneous_basis_search.schedule_fault_search.animate_schedule import (
    ELEMENTS,
    add,
    derive_layers,
)


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "searches/bb64_simultaneous_basis_search" / "schedule_fault_search"
SCHEDULE = DIRECTORY / "baseline_schedule.json"
ANIMATION = DIRECTORY / "planar_schedule_animation.txt"


def load_schedule() -> dict:
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def test_direct_coordinates_are_a_group_isomorphism() -> None:
    images = {to_planar(element) for element in ELEMENTS}
    assert images == {(uu, vv) for uu in range(8) for vv in range(4)}
    for left in ELEMENTS:
        for right in ELEMENTS:
            assert to_planar(add(left, right)) == planar_add(
                to_planar(left), to_planar(right)
            )


def test_all_512_gates_reach_their_planar_ancilla_sites() -> None:
    layers = derive_layers(load_schedule())
    for layer in layers:
        for panel in (layer.x_panel, layer.z_panel):
            motion = to_planar(panel.motion)
            for check, data in enumerate(panel.data_by_check):
                check_coordinate = to_planar(ELEMENTS[check])
                data_coordinate = to_planar(ELEMENTS[data])
                assert planar_add(data_coordinate, motion) == check_coordinate


def test_planar_frames_show_every_atom_and_exactly_64_bonds() -> None:
    schedule = load_schedule()
    previous = initial_placements()
    for layer in derive_layers(schedule):
        changes = transitions(previous, layer)
        position = render_position(layer, changes, schedule["schedule_id"])
        contact = render_contact(layer, changes, schedule["schedule_id"])
        position_grid = "\n".join(
            line for line in position.splitlines() if line.startswith("v=")
        )
        contact_grid = "\n".join(
            line for line in contact.splitlines() if line.startswith("v=")
        )
        assert "--" not in position_grid
        assert contact_grid.count("--") == 64
        for uu in range(8):
            for vv in range(4):
                assert contact_grid.count(f"XZ{uu}{vv}") == 1
                assert contact_grid.count(f"L{uu}{vv}") == 1
                assert contact_grid.count(f"R{uu}{vv}") == 1
        previous = placements(layer)


def test_first_planar_wrap_and_second_side_exchange_are_explicit() -> None:
    schedule = load_schedule()
    layers = derive_layers(schedule)
    first_changes = transitions(initial_placements(), layers[0])
    first = render_contact(layers[0], first_changes, schedule["schedule_id"])
    assert " L00--XZ00-- R10" in first
    assert "*R00" in first
    assert len(first_changes[0].wrapped) == 0
    assert len(first_changes[1].wrapped) == 4

    second_changes = transitions(placements(layers[0]), layers[1])
    second = render_position(layers[1], second_changes, schedule["schedule_id"])
    assert "L: left/X -> right/Z" in second
    assert "upper exterior corridor" in second
    assert "R: right/Z -> left/X" in second
    assert "lower exterior corridor" in second


def test_saved_planar_animation_matches_schedule() -> None:
    schedule = load_schedule()
    layers = derive_layers(schedule)
    rendered = render_all(layers, schedule["schedule_id"])
    rendered.encode("ascii")
    assert len(animation_frames(layers, schedule["schedule_id"])) == 18
    assert rendered.count("64 bonds ON") == 8
    assert ANIMATION.read_text(encoding="ascii") == rendered
