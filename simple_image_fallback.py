"""Small standard-library image feature fallback for self-read.

This is not meant to compete with Pillow. It gives Ina a local decoder for a
few simple image formats when optional media dependencies are unavailable.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple


class ImageFallbackError(ValueError):
    """Raised when the fallback decoder cannot handle an image."""


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_source(source: Any) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()

    if not hasattr(source, "read"):
        raise ImageFallbackError("image source is not readable")

    stream = source  # type: BinaryIO
    try:
        position = stream.tell()
    except Exception:
        position = None
    try:
        stream.seek(0)
    except Exception:
        pass
    data = stream.read()
    if position is not None:
        try:
            stream.seek(position)
        except Exception:
            pass
    if not isinstance(data, (bytes, bytearray)):
        raise ImageFallbackError("image source did not return bytes")
    return bytes(data)


def _luma(r: int, g: int, b: int) -> int:
    return max(0, min(255, int(round((0.299 * r) + (0.587 * g) + (0.114 * b)))))


def _sample(values: Sequence[int], limit: int) -> List[int]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return [int(v) for v in values]
    step = float(len(values)) / float(limit)
    return [int(values[min(len(values) - 1, int(i * step))]) for i in range(limit)]


def _png_paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(data: bytes) -> Tuple[str, int, int, List[int]]:
    if not data.startswith(PNG_SIGNATURE):
        raise ImageFallbackError("not a PNG image")

    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    palette: List[Tuple[int, int, int]] = []
    idat_parts: List[bytes] = []

    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_start = offset + 8
        chunk_end = chunk_start + length
        if chunk_end + 4 > len(data):
            raise ImageFallbackError("truncated PNG chunk")
        chunk = data[chunk_start:chunk_end]
        offset = chunk_end + 4

        if chunk_type == b"IHDR":
            if len(chunk) != 13:
                raise ImageFallbackError("invalid PNG header")
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ImageFallbackError("unsupported PNG compression/filter/interlace mode")
            if bit_depth != 8:
                raise ImageFallbackError("fallback PNG decoder only supports 8-bit images")
            if color_type not in {0, 2, 3, 4, 6}:
                raise ImageFallbackError("unsupported PNG color type")
            if width <= 0 or height <= 0 or width * height > 100_000_000:
                raise ImageFallbackError("unsupported PNG dimensions")
        elif chunk_type == b"PLTE":
            palette = [
                (chunk[i], chunk[i + 1], chunk[i + 2])
                for i in range(0, len(chunk) - 2, 3)
            ]
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ImageFallbackError("PNG missing header")
    if not idat_parts:
        raise ImageFallbackError("PNG missing image data")

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    bpp = channels
    stride = width * channels
    raw = zlib.decompress(b"".join(idat_parts))
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise ImageFallbackError("truncated PNG image data")

    rows: List[bytearray] = []
    pos = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        row = bytearray(raw[pos:pos + stride])
        pos += stride
        for i, value in enumerate(row):
            left = row[i - bpp] if i >= bpp else 0
            up = previous[i]
            up_left = previous[i - bpp] if i >= bpp else 0
            if filter_type == 1:
                row[i] = (value + left) & 0xFF
            elif filter_type == 2:
                row[i] = (value + up) & 0xFF
            elif filter_type == 3:
                row[i] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[i] = (value + _png_paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ImageFallbackError("unsupported PNG row filter")
        rows.append(row)
        previous = row

    pixels: List[int] = []
    if color_type == 0:
        for row in rows:
            pixels.extend(row)
    elif color_type == 3:
        if not palette:
            raise ImageFallbackError("indexed PNG missing palette")
        for row in rows:
            for idx in row:
                if idx >= len(palette):
                    raise ImageFallbackError("indexed PNG palette reference out of range")
                pixels.append(_luma(*palette[idx]))
    else:
        for row in rows:
            for i in range(0, len(row), channels):
                if color_type == 4:
                    gray = row[i]
                else:
                    gray = _luma(row[i], row[i + 1], row[i + 2])
                pixels.append(gray)

    return "png", width, height, pixels


def _decode_bmp(data: bytes) -> Tuple[str, int, int, List[int]]:
    if not data.startswith(b"BM") or len(data) < 54:
        raise ImageFallbackError("not a BMP image")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40 or len(data) < 14 + dib_size:
        raise ImageFallbackError("unsupported BMP header")

    width = struct.unpack_from("<i", data, 18)[0]
    raw_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if planes != 1 or compression != 0:
        raise ImageFallbackError("fallback BMP decoder only supports uncompressed BMP")
    if width <= 0 or raw_height == 0 or abs(raw_height) * width > 100_000_000:
        raise ImageFallbackError("unsupported BMP dimensions")
    if bpp not in {8, 24, 32}:
        raise ImageFallbackError("fallback BMP decoder supports 8/24/32-bit BMP")

    height = abs(raw_height)
    top_down = raw_height < 0
    bytes_per_pixel = max(1, bpp // 8)
    row_stride = ((width * bpp + 31) // 32) * 4
    if pixel_offset + (row_stride * height) > len(data):
        raise ImageFallbackError("truncated BMP pixel data")

    palette: List[Tuple[int, int, int]] = []
    if bpp == 8:
        palette_bytes = data[14 + dib_size:pixel_offset]
        for i in range(0, len(palette_bytes) - 3, 4):
            b, g, r, _ = palette_bytes[i:i + 4]
            palette.append((r, g, b))
        if not palette:
            raise ImageFallbackError("8-bit BMP missing palette")

    pixels: List[int] = []
    row_range = range(height) if top_down else range(height - 1, -1, -1)
    for row_index in row_range:
        row_start = pixel_offset + (row_index * row_stride)
        row = data[row_start:row_start + row_stride]
        for x in range(width):
            pos = x * bytes_per_pixel
            if bpp == 8:
                idx = row[pos]
                if idx >= len(palette):
                    raise ImageFallbackError("BMP palette reference out of range")
                pixels.append(_luma(*palette[idx]))
            else:
                b, g, r = row[pos], row[pos + 1], row[pos + 2]
                pixels.append(_luma(r, g, b))

    return "bmp", width, height, pixels


def _pnm_token(data: bytes, pos: int) -> Tuple[Optional[bytes], int]:
    length = len(data)
    while pos < length:
        value = data[pos]
        if value == 35:  # #
            while pos < length and data[pos] not in b"\r\n":
                pos += 1
        elif chr(value).isspace():
            pos += 1
        else:
            break
    if pos >= length:
        return None, pos
    start = pos
    while pos < length and not chr(data[pos]).isspace():
        pos += 1
    return data[start:pos], pos


def _decode_pnm(data: bytes) -> Tuple[str, int, int, List[int]]:
    magic, pos = _pnm_token(data, 0)
    if magic not in {b"P2", b"P3", b"P5", b"P6"}:
        raise ImageFallbackError("not a supported PNM image")
    width_token, pos = _pnm_token(data, pos)
    height_token, pos = _pnm_token(data, pos)
    max_token, pos = _pnm_token(data, pos)
    if not width_token or not height_token or not max_token:
        raise ImageFallbackError("PNM header is incomplete")
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_token)
    if width <= 0 or height <= 0 or width * height > 100_000_000:
        raise ImageFallbackError("unsupported PNM dimensions")
    if max_value <= 0 or max_value > 255:
        raise ImageFallbackError("fallback PNM decoder only supports 8-bit samples")

    while pos < len(data) and chr(data[pos]).isspace():
        pos += 1

    scale = 255.0 / float(max_value)
    pixels: List[int] = []
    if magic in {b"P2", b"P3"}:
        values: List[int] = []
        while True:
            token, pos = _pnm_token(data, pos)
            if token is None:
                break
            values.append(int(token))
        if magic == b"P2":
            if len(values) < width * height:
                raise ImageFallbackError("truncated PGM image data")
            pixels = [int(round(v * scale)) for v in values[:width * height]]
        else:
            if len(values) < width * height * 3:
                raise ImageFallbackError("truncated PPM image data")
            for i in range(0, width * height * 3, 3):
                pixels.append(_luma(
                    int(round(values[i] * scale)),
                    int(round(values[i + 1] * scale)),
                    int(round(values[i + 2] * scale)),
                ))
    elif magic == b"P5":
        expected = width * height
        raw = data[pos:pos + expected]
        if len(raw) < expected:
            raise ImageFallbackError("truncated PGM image data")
        pixels = [int(round(v * scale)) for v in raw]
    else:
        expected = width * height * 3
        raw = data[pos:pos + expected]
        if len(raw) < expected:
            raise ImageFallbackError("truncated PPM image data")
        for i in range(0, expected, 3):
            pixels.append(_luma(
                int(round(raw[i] * scale)),
                int(round(raw[i + 1] * scale)),
                int(round(raw[i + 2] * scale)),
            ))

    return magic.decode("ascii").lower(), width, height, pixels


def _decode(data: bytes) -> Tuple[str, int, int, List[int]]:
    if data.startswith(PNG_SIGNATURE):
        return _decode_png(data)
    if data.startswith(b"BM"):
        return _decode_bmp(data)
    if data[:2] in {b"P2", b"P3", b"P5", b"P6"}:
        return _decode_pnm(data)
    raise ImageFallbackError("stdlib fallback supports PNG, BMP, PGM, and PPM images")


def extract_image_grid(
    source: Any,
    *,
    max_width: int = 128,
    max_height: int = 128,
) -> Dict[str, Any]:
    """Decode an image into a bounded spatial grayscale grid.

    Unlike :func:`extract_image_features`, this preserves two-dimensional
    neighbourhoods for visual-token discovery. Downsampling uses a bounded
    set of samples across each source cell so thin forms are less likely to
    disappear, and never returns more than ``max_width * max_height`` pixels.
    """

    data = _read_source(source)
    image_format, width, height, pixels = _decode(data)
    target_width = max(1, min(int(max_width), width))
    target_height = max(1, min(int(max_height), height))
    scale = min(1.0, target_width / float(width), target_height / float(height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))

    sampled: List[int] = []
    for target_y in range(target_height):
        source_y0 = int(target_y * height / target_height)
        source_y1 = max(source_y0 + 1, int((target_y + 1) * height / target_height))
        y_step = max(1, (source_y1 - source_y0) // 4)
        y_samples = list(range(source_y0, source_y1, y_step))[:4]
        for target_x in range(target_width):
            source_x0 = int(target_x * width / target_width)
            source_x1 = max(source_x0 + 1, int((target_x + 1) * width / target_width))
            x_step = max(1, (source_x1 - source_x0) // 4)
            x_samples = list(range(source_x0, source_x1, x_step))[:4]
            values = [
                pixels[min(height - 1, source_y) * width + min(width - 1, source_x)]
                for source_y in y_samples
                for source_x in x_samples
            ]
            sampled.append(int(round(sum(values) / max(1, len(values)))))

    return {
        "decoder": "simple_image_fallback",
        "format": image_format,
        "source_width": width,
        "source_height": height,
        "width": target_width,
        "height": target_height,
        "pixels": sampled,
    }


def extract_image_features(source: Any, *, limit: int = 1024) -> Dict[str, Any]:
    """Decode simple image formats into grayscale feature values.

    The returned feature list intentionally uses 0-255 grayscale values so it
    matches the old Pillow/NumPy path closely enough for the transformer.
    """

    data = _read_source(source)
    image_format, width, height, pixels = _decode(data)
    return {
        "decoder": "simple_image_fallback",
        "format": image_format,
        "width": width,
        "height": height,
        "features": _sample(pixels, int(limit)),
        "feature_count": min(len(pixels), int(limit)),
        "source_pixels": len(pixels),
    }


__all__ = ["ImageFallbackError", "extract_image_features", "extract_image_grid"]
