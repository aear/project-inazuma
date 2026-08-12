"""Dependency-free numeric containers for media and ML boundary code."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RGBFrame:
    """Packed RGB framebuffer with bounded nearest-neighbour downsampling."""

    width: int
    height: int
    data: bytes

    def __post_init__(self) -> None:
        width = int(self.width)
        height = int(self.height)
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if len(self.data) != width * height * 3:
            raise ValueError("RGB payload length does not match frame dimensions")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    def tobytes(self) -> bytes:
        return self.data

    def downsample(self, step: int) -> "RGBFrame":
        step = max(1, int(step))
        if step == 1:
            return self
        width = (self.width + step - 1) // step
        height = (self.height + step - 1) // step
        reduced = bytearray(width * height * 3)
        target = 0
        for y in range(0, self.height, step):
            row = y * self.width * 3
            for x in range(0, self.width, step):
                source = row + x * 3
                reduced[target:target + 3] = self.data[source:source + 3]
                target += 3
        return RGBFrame(width, height, bytes(reduced))

    def fit(self, max_width: int, max_height: int) -> "RGBFrame":
        max_width = max(1, int(max_width))
        max_height = max(1, int(max_height))
        step = max(
            1,
            (self.width + max_width - 1) // max_width,
            (self.height + max_height - 1) // max_height,
        )
        return self.downsample(step)

    def ppm(self, *, max_width: int | None = None, max_height: int | None = None) -> bytes:
        frame = self
        if max_width is not None or max_height is not None:
            frame = self.fit(max_width or self.width, max_height or self.height)
        header = f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii")
        return header + frame.data
