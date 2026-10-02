"""Choose and certify balanced three-part TMR decompositions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .model import CodeArtifact, gf2_rank, vector_from_support


Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PartitionCertificate:
    partitions: dict[int, Partition]
    batch_ranks: tuple[int, ...]
    combined_rank: int
    batch_kernel_dimensions: tuple[int, ...]
    combined_kernel_dimension: int
    single_final_check_certified: bool

    def to_json(self) -> dict[str, object]:
        return {
            "partitions": {
                str(logical): [list(part) for part in partition]
                for logical, partition in sorted(self.partitions.items())
            },
            "batch_ranks": list(self.batch_ranks),
            "combined_rank": self.combined_rank,
            "batch_kernel_dimensions": list(self.batch_kernel_dimensions),
            "combined_kernel_dimension": self.combined_kernel_dimension,
            "single_final_check_certified": self.single_final_check_certified,
        }


def deterministic_balanced_partitions(code: CodeArtifact) -> dict[int, Partition]:
    """Split each sorted weight-seven support as ``2+2+3``."""

    result: dict[int, Partition] = {}
    for logical, row in enumerate(code.logical_z):
        support = tuple(np.flatnonzero(row).astype(int).tolist())
        result[logical] = (support[:2], support[2:4], support[4:])
    return result


def syndrome_matrix(
    code: CodeArtifact,
    partitions: Mapping[int, Partition],
    logicals: Iterable[int],
) -> np.ndarray:
    columns = []
    for logical in logicals:
        for part in partitions[int(logical)]:
            columns.append(
                (code.matrix_x @ vector_from_support(part, code.num_data)) % 2
            )
    return np.stack(columns, axis=1)


def _validate_partition_supports(
    code: CodeArtifact, partitions: Mapping[int, Partition]
) -> None:
    if set(partitions) != set(range(code.num_logicals)):
        raise ValueError(
            f"partitions must cover logicals 0 through {code.num_logicals - 1}"
        )
    for logical, parts in partitions.items():
        if len(parts) != 3 or sorted(map(len, parts)) != [2, 2, 3]:
            raise ValueError(f"logical {logical} is not partitioned as 2+2+3")
        flattened = [qubit for part in parts for qubit in part]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"logical {logical} partition pieces overlap")
        expected = set(np.flatnonzero(code.logical_z[logical]).astype(int).tolist())
        if set(flattened) != expected:
            raise ValueError(f"logical {logical} partition has the wrong union")
        # Distance six implies every proper nonempty subset has nonzero
        # syndrome, but check this explicitly rather than relying on metadata.
        for mask in range(1, 2 ** len(parts) - 1):
            support: set[int] = set()
            for index, part in enumerate(parts):
                if mask & (1 << index):
                    support.symmetric_difference_update(part)
            syndrome = (
                code.matrix_x @ vector_from_support(support, code.num_data)
            ) % 2
            if not np.any(syndrome):
                raise ValueError(
                    f"logical {logical} has an undetected proper partition product"
                )


def certify_partitions(
    code: CodeArtifact,
    partitions: Mapping[int, Partition],
) -> PartitionCertificate:
    _validate_partition_supports(code, partitions)
    ranks = tuple(
        gf2_rank(syndrome_matrix(code, partitions, batch)) for batch in code.batches
    )
    combined_rank = gf2_rank(
        syndrome_matrix(code, partitions, range(code.num_logicals))
    )
    expected_batch_ranks = tuple(2 * len(batch) for batch in code.batches)
    expected_combined_rank = 2 * code.num_logicals
    certificate = PartitionCertificate(
        partitions={logical: tuple(tuple(part) for part in parts) for logical, parts in partitions.items()},
        batch_ranks=tuple(map(int, ranks)),
        combined_rank=int(combined_rank),
        batch_kernel_dimensions=tuple(
            3 * len(batch) - int(rank)
            for batch, rank in zip(code.batches, ranks, strict=True)
        ),
        combined_kernel_dimension=3 * code.num_logicals - int(combined_rank),
        single_final_check_certified=(
            ranks == expected_batch_ranks and combined_rank == expected_combined_rank
        ),
    )
    if certificate.batch_ranks != expected_batch_ranks:
        raise ValueError(f"batch TMR kernels contain unintended terms: {certificate}")
    return certificate


def save_certificate(path: Path, certificate: PartitionCertificate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(certificate.to_json(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_partitions(path: Path) -> dict[int, Partition]:
    with Path(path).open(encoding="utf-8") as source:
        record = json.load(source)
    payload = record.get("partitions", record)
    return {
        int(logical): tuple(tuple(map(int, part)) for part in parts)
        for logical, parts in payload.items()
    }
