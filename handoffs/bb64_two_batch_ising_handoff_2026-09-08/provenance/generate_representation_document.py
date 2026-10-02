#!/usr/bin/env python3
"""Generate the self-contained preferred two-batch BB64 specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from search_helpers import (
    batch_disjointness,
    gf2_rank,
    permutation_matrix_test,
    permutation_order,
)


DIRECTORY = Path(__file__).resolve().parent
PACKAGE_DIRECTORY = DIRECTORY.parent
DEFAULT_BASIS = PACKAGE_DIRECTORY / "bb64_two_batch_basis.npz"
DEFAULT_SCHEDULE = PACKAGE_DIRECTORY / "baseline_schedule.json"
DEFAULT_OUTPUT = PACKAGE_DIRECTORY / "BB64_TWO_BATCH_REPRESENTATION.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def support(row: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(row, dtype=np.uint8)).astype(int).tolist()


def tex_set(values: Iterable[int]) -> str:
    return r"\{" + ",".join(map(str, values)) + r"\}"


def batches(labels: np.ndarray) -> list[list[int]]:
    return [
        np.flatnonzero(labels == label).astype(int).tolist()
        for label in sorted(set(map(int, labels)))
    ]


def permutation_cycle_list(
    permutation: Sequence[int], *, prefix: str = ""
) -> list[str]:
    seen: set[int] = set()
    cycles: list[str] = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle = []
        value = start
        while value not in seen:
            seen.add(value)
            cycle.append(value)
            value = int(permutation[value])
        if len(cycle) > 1:
            labels = (f"{prefix}_{{{item}}}" if prefix else str(item) for item in cycle)
            cycles.append("(" + r"\;".join(labels) + ")")
    return cycles


def permutation_cycles(permutation: Sequence[int]) -> str:
    return "".join(permutation_cycle_list(permutation)) or "()"


def aligned_cycle_decomposition(
    symbol: str,
    permutation: Sequence[int],
    *,
    cycles_per_line: int,
    prefix: str = "",
) -> str:
    cycles = permutation_cycle_list(permutation, prefix=prefix)
    chunks = [
        "".join(cycles[start : start + cycles_per_line])
        for start in range(0, len(cycles), cycles_per_line)
    ]
    if not chunks:
        return rf"{symbol}&=()"
    lines = [rf"{symbol}&={chunks[0]}"]
    lines.extend(rf"&\quad {chunk}" for chunk in chunks[1:])
    return r",\\".join(lines)


def displayed_check_row_map(check: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(check)
    transformed[:, permutation] = check
    mapping = []
    for row in transformed:
        matches = np.flatnonzero(np.all(check == row, axis=1))
        if len(matches) != 1:
            raise ValueError("automorphism does not uniquely permute displayed check rows")
        mapping.append(int(matches[0]))
    return np.asarray(mapping, dtype=np.int64)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a pipe table whose separators align in the Markdown source."""
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("table row has the wrong number of columns")
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows), 3)
        for column in range(len(headers))
    ]

    def render(row: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join((render(headers), separator, *(render(row) for row in rows)))


def physical_label_table() -> str:
    headers = ["Half and first coordinate", *(f"`y={yy}`" for yy in range(8))]
    rows: list[list[str]] = []
    for half, offset in (("L", 0), ("R", 32)):
        for xx in range(4):
            entries = [f"`q{offset + 8 * xx + yy}`" for yy in range(8)]
            rows.append([f"`{half}, x={xx}`", *entries])
    return markdown_table(headers, rows)


def logical_table(logical_z: np.ndarray, logical_x: np.ndarray, p: np.ndarray) -> str:
    rows: list[list[str]] = []
    for logical in range(8):
        z_labels = "{" + ", ".join(f"q{qubit}" for qubit in support(logical_z[logical])) + "}"
        x_labels = "{" + ", ".join(f"q{qubit}" for qubit in support(logical_x[logical])) + "}"
        rows.append(
            [
                f"`{logical}`",
                f"`{z_labels}`",
                f"`{x_labels}`",
                f"`Z{logical} -> X{int(p[logical])}`",
            ]
        )
    return markdown_table(
        ("Logical", "Z support", "X support", "After transversal H"), rows
    )


def _schedule_pairs(layer: dict[str, Any], kind: str) -> list[tuple[int, int]]:
    gates = [gate for gate in layer["gates"] if gate["type"] == kind]
    pairs = sorted((int(gate["check"]), int(gate["data"])) for gate in gates)
    if [check for check, _data in pairs] != list(range(32)):
        raise ValueError(f"layer does not use every {kind} ancilla exactly once")
    return pairs


def _mapping(kind: str, check: int, data: int) -> str:
    if kind == "X":
        return rf"x_{{{check}}}\!\to q_{{{data}}}"
    return rf"q_{{{data}}}\!\to z_{{{check}}}"


def schedule_layer_tex(layer_index: int, layer: dict[str, Any]) -> str:
    lines = [f"### CNOT layer {layer_index}", "", r"\["]
    lines.append(r"\begin{aligned}")
    for kind in ("X", "Z"):
        pairs = _schedule_pairs(layer, kind)
        for chunk_index in range(4):
            chunk = pairs[8 * chunk_index : 8 * (chunk_index + 1)]
            mappings = r",\quad ".join(
                _mapping(kind, check, data) for check, data in chunk
            )
            label = rf"\mathrm{{{kind}}}:\quad " if chunk_index == 0 else r"\phantom{\mathrm{X}:\quad}"
            ending = r",\\" if chunk_index < 3 or kind == "X" else r"."
            lines.append(f"{label}&{mappings}{ending}")
        if kind == "X":
            lines.append(r"\\[-2pt]")
    lines.extend((r"\end{aligned}", r"\]", ""))
    return "\n".join(lines)


def validate(
    archive: Any,
    schedule: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[list[int]]]:
    check_x = np.asarray(archive["matrix_x"], dtype=np.uint8)
    check_z = np.asarray(archive["matrix_z"], dtype=np.uint8)
    logical_z = np.asarray(archive["logical_z"], dtype=np.uint8)
    logical_x = np.asarray(archive["logical_x"], dtype=np.uint8)
    p = np.asarray(archive["hadamard_permutation"], dtype=np.int64)
    z_batches = batches(np.asarray(archive["disjoint_batch_of_z_logical"], dtype=np.int64))
    x_batches = batches(np.asarray(archive["disjoint_batch_of_x_logical"], dtype=np.int64))

    if not np.array_equal(check_x, check_z):
        raise ValueError("basis is not in the claimed self-dual presentation")
    if check_x.shape != (32, 64) or gf2_rank(check_x) != 28:
        raise ValueError("unexpected check matrix shape or rank")
    if set(np.count_nonzero(check_x, axis=1).tolist()) != {8}:
        raise ValueError("checks are not uniformly weight eight")
    if logical_z.shape != logical_x.shape or logical_z.shape != (8, 64):
        raise ValueError("unexpected logical basis shape")
    if set(np.count_nonzero(logical_z, axis=1).tolist()) != {8}:
        raise ValueError("Z logicals are not uniformly weight eight")
    if not np.array_equal(logical_x, logical_z[p]):
        raise ValueError("saved X supports do not realize the exact H permutation")
    if not np.array_equal((logical_z @ logical_x.T) % 2, np.eye(8, dtype=np.uint8)):
        raise ValueError("logical basis is not canonical")
    if not permutation_matrix_test((logical_z @ logical_z.T) % 2):
        raise ValueError("physical H does not have permutation ZX pairing")
    if not batch_disjointness(logical_z, z_batches):
        raise ValueError("saved Z batches are not internally disjoint")
    if not batch_disjointness(logical_x, x_batches):
        raise ValueError("saved X batches are not internally disjoint")
    if schedule.get("schedule_id") != "67e327dc7cb35335":
        raise ValueError("unexpected syndrome schedule")
    if schedule.get("cnot_depth") != 8 or len(schedule.get("layers", [])) != 8:
        raise ValueError("schedule is not depth eight")
    if schedule.get("cnot_count") != 512:
        raise ValueError("schedule does not contain 512 CNOTs")
    scheduled_edges: dict[str, set[tuple[int, int]]] = {"X": set(), "Z": set()}
    for layer in schedule["layers"]:
        layer_data: list[int] = []
        for kind in ("X", "Z"):
            pairs = _schedule_pairs(layer, kind)
            scheduled_edges[kind].update(pairs)
            layer_data.extend(data for _check, data in pairs)
        if sorted(layer_data) != list(range(64)):
            raise ValueError("a CNOT layer is not a perfect matching on the data")
    for kind, matrix in (("X", check_x), ("Z", check_z)):
        expected = {
            (check, int(data))
            for check, row in enumerate(matrix)
            for data in np.flatnonzero(row)
        }
        if scheduled_edges[kind] != expected:
            raise ValueError(f"schedule does not implement the saved {kind} checks")
    return logical_z, logical_x, p, z_batches, x_batches


def render_document(basis_path: Path, schedule_path: Path) -> str:
    archive = np.load(basis_path)
    schedule = json.loads(schedule_path.read_text())
    logical_z, logical_x, p, z_batches, x_batches = validate(archive, schedule)
    grid_x = np.argmax(
        np.asarray(archive["grid_x_z_logical_action"], dtype=np.uint8), axis=1
    )
    grid_y = np.argmax(
        np.asarray(archive["grid_y_z_logical_action"], dtype=np.uint8), axis=1
    )
    physical_x = np.asarray(archive["grid_x_physical_permutation"], dtype=np.int64)
    physical_y = np.asarray(archive["grid_y_physical_permutation"], dtype=np.int64)
    check = np.asarray(archive["matrix_x"], dtype=np.uint8)
    if permutation_order(physical_x) != 8 or permutation_order(physical_y) != 2:
        raise ValueError("unexpected physical automorphism order")
    if not np.array_equal(physical_x[physical_y], physical_y[physical_x]):
        raise ValueError("physical automorphism generators do not commute")
    check_x_map = displayed_check_row_map(check, physical_x)
    check_y_map = displayed_check_row_map(check, physical_y)

    z_union = [
        support(np.any(logical_z[np.asarray(batch, dtype=np.int64)], axis=0))
        for batch in z_batches
    ]
    schedule_sections = "\n".join(
        schedule_layer_tex(layer_index, layer)
        for layer_index, layer in enumerate(schedule["layers"])
    )
    return rf"""# Preferred two-batch representation of the `[[64,8,8]]` BB code

This document fixes one complete physical presentation of the Liang--Chen
self-dual bivariate-bicycle code used for the `C_4 x C_2` Ising block.  All
indices are zero based.  It was generated from the preferred candidate-0 basis
and the canonical simultaneous syndrome schedule.

## Source identity

- Basis SHA-256: `{sha256(basis_path)}`
- Schedule SHA-256: `{sha256(schedule_path)}`
- Schedule ID: `{schedule['schedule_id']}`
- Code parameters: `[[64,8,8]]`
- Check presentation: $H_X=H_Z$, with 32 displayed rows of each type, rank 28,
  and row weight 8.

## Physical data-qubit labels

The data qubits are $q_0,q_1,\ldots,q_{{63}}$.  The first BB half is
$L=\{{q_0,\ldots,q_{{31}}\}}$ and the second is
$R=\{{q_{{32}},\ldots,q_{{63}}\}}$.  In the stored twisted-torus coordinates,

\[
q(L,x,y)=q_{{8x+y}},\qquad q(R,x,y)=q_{{32+8x+y}},
\qquad x\in\mathbb Z_4,\ y\in\mathbb Z_8.
\]

{physical_label_table()}

These coordinates label the code artifact.  They do not assert that the
neutral-atom array must be embedded with this twisted periodic boundary in the
laboratory plane.

## Logical X and Z representatives

For a binary support $S\subseteq\{{0,\ldots,63\}}$, define

\[
Z(S)=\prod_{{j\in S}} Z_{{q_j}},\qquad
X(S)=\prod_{{j\in S}} X_{{q_j}}.
\]

The saved physical representatives are:

{logical_table(logical_z, logical_x, p)}

Every displayed logical representative has physical weight eight.  The eight
pairs are canonical:

\[
\bar Z_i\bar X_j=(-1)^{{\delta_{{ij}}}}\bar X_j\bar Z_i.
\]

## Disjoint logical-Z batches

The eight Z logicals are covered exactly once by two batches:

\[
\mathcal B_Z^{{(0)}}={tex_set(z_batches[0])},\qquad
\mathcal B_Z^{{(1)}}={tex_set(z_batches[1])}.
\]

Their union supports are

\[
\begin{{aligned}}
\operatorname{{supp}}\!\left(\mathcal B_Z^{{(0)}}\right)
  &={tex_set(z_union[0])},\\
\operatorname{{supp}}\!\left(\mathcal B_Z^{{(1)}}\right)
  &={tex_set(z_union[1])}.
\end{{aligned}}
\]

Each union contains 32 qubits because the four weight-eight logical supports
inside that batch are pairwise disjoint.  Supports in different batches are
allowed to overlap, which is why the batches are executed separately.

For completeness, the corresponding internally disjoint X batches are

\[
\mathcal B_X^{{(0)}}={tex_set(x_batches[0])},\qquad
\mathcal B_X^{{(1)}}={tex_set(x_batches[1])}.
\]

## Transversal Hadamard and logical permutations

The exact support identity is

\[
\operatorname{{supp}}(\bar X_i)=\operatorname{{supp}}(\bar Z_{{p(i)}}),
\qquad
p={permutation_cycles(p)}.
\]

Consequently,

\[
H^{{\otimes64}}:\quad
\bar Z_i\longmapsto\bar X_{{p(i)}},\qquad
\bar X_i\longmapsto\bar Z_{{p(i)}}.
\]

The two physical translation generators induce the commuting logical
permutations

\[
T_x={permutation_cycles(grid_x)},\qquad
T_y={permutation_cycles(grid_y)}.
\]

Thus the logical indices form a regular `C_4 x C_2` orbit.

## Explicit physical automorphisms

Let $U_P$ denote the unitary that moves the quantum state at physical data
site $q_j$ to site $q_{{P(j)}}$.  The convention used here is

\[
U_P Z_{{q_j}}U_P^\dagger=Z_{{q_{{P(j)}}}},\qquad
U_P X_{{q_j}}U_P^\dagger=X_{{q_{{P(j)}}}}.
\]

### Physical lift of the logical `C_4` generator

The physical permutation $P_x$ is

\[
\begin{{aligned}}
{aligned_cycle_decomposition('P_x', physical_x, cycles_per_line=2, prefix='q')}
\end{{aligned}}
\]

Equivalently, on the strict planar `C_8 x C_4` data array,

\[
P_x:\quad L(u,v)\mapsto L(u+1,v),\qquad
R(u,v)\mapsto R(u+1,v).
\]

Its physical order is eight, although its induced logical order is four.  The
kernel element $P_x^4$ shifts every atom by four planar columns while acting
trivially on the logical subsystem.

### Physical lift of the logical `C_2` generator

The physical permutation $P_y$ is the following set of 32 simultaneous swaps:

\[
\begin{{aligned}}
{aligned_cycle_decomposition('P_y', physical_y, cycles_per_line=4, prefix='q')}
\end{{aligned}}
\]

In strict planar coordinates its endpoint map is

\[
P_y:\quad L(u,v)\mapsto R(u-2v,-v),\qquad
R(u,v)\mapsto L(u-2v,-v),
\]

where the first coordinate is modulo eight and the second is modulo four.
Thus $P_y$ exchanges the two BB data halves, reflects the row coordinate, and
applies a row-dependent horizontal displacement.  Its physical and logical
orders are both two.

The physical generators commute.  They generate a physical
$C_8\times C_2$ action whose logical quotient by the logically trivial
$\langle P_x^4\rangle$ action is the desired $C_4\times C_2$ translation
group.

Both permutations map each displayed X-check and Z-check row directly to
another displayed row, rather than merely preserving the check row space.  If
$s_a^X$ and $s_a^Z$ denote displayed check rows with $a=0,\ldots,31$, their
common row-label permutations are

\[
\begin{{aligned}}
{aligned_cycle_decomposition('r_x', check_x_map, cycles_per_line=2)}\\[2pt]
{aligned_cycle_decomposition('r_y', check_y_map, cycles_per_line=4)}.
\end{{aligned}}
\]

Fixed check-row labels are omitted from this cycle notation; in particular,
$r_y$ fixes rows $0,\ldots,7$.

These equations specify the endpoint permutations exactly.  They certify code
automorphisms, but they do not by themselves provide collision-free continuous
optical-tweezer trajectories between the endpoints.

## Simultaneous X/Z syndrome schedule

There are 32 X ancillas $x_0,\ldots,x_{{31}}$ and 32 Z ancillas
$z_0,\ldots,z_{{31}}$.  For each syndrome round:

1. prepare every $x_a$ in $|+\rangle$ and every $z_a$ in $|0\rangle$;
2. execute the eight CNOT layers below in order;
3. measure every $x_a$ in the X basis and every $z_a$ in the Z basis.

The arrow gives the CNOT control-to-target direction:

\[
x_a\to q_j\quad\text{{for an X-check interaction}},\qquad
q_j\to z_a\quad\text{{for a Z-check interaction}}.
\]

Every layer contains 32 X-check CNOTs and 32 Z-check CNOTs.  Each data qubit
appears exactly once in each layer, so all 64 interactions occur in parallel.

{schedule_sections}
## Schedule interpretation

Across all eight layers, ancilla $x_a$ touches the eight data qubits in X-check
row $a$, while ancilla $z_a$ touches the eight data qubits in Z-check row $a$.
The X and Z row supports are identical because $H_X=H_Z$, but their CNOT
directions and their layer orders differ.  The displayed ordering is the
translation-symmetric, collision-free schedule with clean cross-ancilla
backaction and schedule ID `{schedule['schedule_id']}`.
"""


def main() -> None:
    args = parse_args()
    rendered = render_document(args.basis, args.schedule)
    args.output.write_text(rendered)
    print(
        json.dumps(
            {
                "basis": str(args.basis.resolve()),
                "schedule": str(args.schedule.resolve()),
                "output": str(args.output.resolve()),
                "bytes": len(rendered.encode()),
                "logical_rows": 8,
                "cnot_layers": 8,
                "cnots": 512,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
