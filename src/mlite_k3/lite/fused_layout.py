"""K3-specific declarations for checkpoint tensors fused on their row axis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedWeight:
    """The packed and scale tensors for one MXFP4 checkpoint weight."""

    packed: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class WeightSegment:
    """One named row range in a K3 checkpoint tensor."""

    name: str
    count: int
    head_dim: int
    replicated: bool = False

    @property
    def rows(self) -> int:
        return self.count * self.head_dim

    def local_rows(self, world_size: int) -> int:
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if world_size == 1 or self.replicated:
            return self.rows
        if self.count % world_size:
            raise ValueError(
                f"{self.name} has {self.count} heads, not divisible by "
                f"world_size={world_size}"
            )
        return (self.count // world_size) * self.head_dim


@dataclass(frozen=True)
class FusedWeightLayout:
    """Minimal K3-only row-axis fusion contract for checkpoint IO."""

    name: str
    segments: tuple[WeightSegment, ...]

    def fuse(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse_ordered(
            tuple((segment.name, tensors[segment.name]) for segment in self.segments)
        )

    def fuse_ordered(self, tensors: Sequence[tuple[str, torch.Tensor]]) -> torch.Tensor:
        expected = tuple(segment.name for segment in self.segments)
        actual = tuple(name for name, _ in tensors)
        if actual != expected:
            raise ValueError(
                f"{self.name} segment order mismatch: expected {expected}, got {actual}"
            )
        values: list[torch.Tensor] = []
        for segment, (_, tensor) in zip(self.segments, tensors, strict=True):
            if tensor.ndim < 1 or tensor.size(0) != segment.rows:
                rows = tensor.size(0) if tensor.ndim else 0
                raise ValueError(
                    f"{segment.name} requires {segment.rows} rows, got {rows}"
                )
            values.append(tensor)
        return torch.cat(values, dim=0)

    def split(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_rows = sum(segment.rows for segment in self.segments)
        rows = tensor.size(0) if tensor.ndim else 0
        if tensor.ndim < 1 or rows != expected_rows:
            raise ValueError(
                f"{self.name} requires {expected_rows} fused rows, got {rows}"
            )
        pieces = tensor.split(tuple(segment.rows for segment in self.segments), dim=0)
        return dict(
            zip((segment.name for segment in self.segments), pieces, strict=True)
        )

    def split_quantized(
        self, packed: torch.Tensor, scale: torch.Tensor
    ) -> dict[str, QuantizedWeight]:
        packed_parts = self.split(packed)
        scale_parts = self.split(scale)
        return {
            segment.name: QuantizedWeight(
                packed_parts[segment.name], scale_parts[segment.name]
            )
            for segment in self.segments
        }

    def fuse_quantized_ordered(
        self,
        tensors: Sequence[tuple[str, QuantizedWeight]],
        *,
        materialize: Callable[[QuantizedWeight], torch.Tensor],
    ) -> torch.Tensor:
        return self.fuse_ordered(
            tuple((name, materialize(weight)) for name, weight in tensors)
        )
