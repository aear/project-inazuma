"""Direct X11 capture, input and lightweight window layout via ctypes."""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ina_ml import RGBFrame

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


ZPIXMAP = 2
ALL_PLANES = ctypes.c_ulong(-1).value


class XImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int), ("format", ctypes.c_int),
        ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong), ("obdata", ctypes.c_void_p),
        ("funcs", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    title: str


def _mask_shift(mask: int) -> int:
    shift = 0
    while mask and not (mask & 1):
        mask >>= 1
        shift += 1
    return shift


def _mask_bits(mask: int) -> int:
    return int(mask >> _mask_shift(mask)).bit_length() if mask else 0


class X11Desktop:
    def __init__(self, display: str) -> None:
        x11_name = ctypes.util.find_library("X11") or "libX11.so.6"
        xtst_name = ctypes.util.find_library("Xtst") or "libXtst.so.6"
        self.x11 = ctypes.CDLL(x11_name)
        self.xtst = ctypes.CDLL(xtst_name)
        self._configure_functions()
        self.display_name = str(display)
        self.display = self.x11.XOpenDisplay(self.display_name.encode("utf-8"))
        if not self.display:
            raise RuntimeError(f"cannot open X display {self.display_name}")
        self.root = int(self.x11.XDefaultRootWindow(self.display))
        self._lock = threading.RLock()

    def _configure_functions(self) -> None:
        display_p = ctypes.c_void_p
        self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x11.XOpenDisplay.restype = display_p
        self.x11.XCloseDisplay.argtypes = [display_p]
        self.x11.XDefaultRootWindow.argtypes = [display_p]
        self.x11.XDefaultRootWindow.restype = ctypes.c_ulong
        self.x11.XGetGeometry.argtypes = [
            display_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
        ]
        self.x11.XGetImage.argtypes = [
            display_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int,
        ]
        self.x11.XGetImage.restype = ctypes.POINTER(XImage)
        self.x11.XDestroyImage.argtypes = [ctypes.POINTER(XImage)]
        self.x11.XQueryTree.argtypes = [
            display_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.x11.XFetchName.argtypes = [display_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
        self.x11.XFree.argtypes = [ctypes.c_void_p]
        self.x11.XMoveResizeWindow.argtypes = [
            display_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
        ]
        self.x11.XRaiseWindow.argtypes = [display_p, ctypes.c_ulong]
        self.x11.XSetInputFocus.argtypes = [display_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        self.x11.XGetInputFocus.argtypes = [
            display_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
        ]
        self.x11.XGetInputFocus.restype = ctypes.c_int
        self.x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self.x11.XStringToKeysym.restype = ctypes.c_ulong
        self.x11.XKeysymToKeycode.argtypes = [display_p, ctypes.c_ulong]
        self.x11.XKeysymToKeycode.restype = ctypes.c_uint
        self.x11.XFlush.argtypes = [display_p]
        self.xtst.XTestFakeMotionEvent.argtypes = [display_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
        self.xtst.XTestFakeButtonEvent.argtypes = [display_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self.xtst.XTestFakeKeyEvent.argtypes = [display_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    def close(self) -> None:
        with self._lock:
            if self.display:
                self.x11.XCloseDisplay(self.display)
                self.display = None

    def size(self) -> tuple[int, int]:
        root = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        with self._lock:
            ok = self.x11.XGetGeometry(
                self.display, self.root, ctypes.byref(root), ctypes.byref(x), ctypes.byref(y),
                ctypes.byref(width), ctypes.byref(height), ctypes.byref(border), ctypes.byref(depth),
            )
        if not ok:
            raise RuntimeError("could not read virtual desktop geometry")
        return int(width.value), int(height.value)

    def _capture_rgb(self) -> RGBFrame:
        width, height = self.size()
        with self._lock:
            image = self.x11.XGetImage(
                self.display, self.root, 0, 0, width, height, ALL_PLANES, ZPIXMAP
            )
            if not image:
                raise RuntimeError("XGetImage failed")
            try:
                meta = image.contents
                raw = ctypes.string_at(meta.data, meta.bytes_per_line * meta.height)
                bytes_per_pixel = max(1, meta.bits_per_pixel // 8)
                if bytes_per_pixel != 4:
                    raise RuntimeError(f"unsupported X image depth: {meta.bits_per_pixel}")
                # Xvfb's standard 24-bit screen is stored as BGRX. Slice assignment
                # performs the channel shuffle in C and avoids a Python pixel loop.
                if (meta.red_mask, meta.green_mask, meta.blue_mask) == (0xFF0000, 0xFF00, 0xFF):
                    packed_bytes = bytearray(width * height * 4)
                    for row in range(height):
                        source = row * meta.bytes_per_line
                        target = row * width * 4
                        packed_bytes[target:target + width * 4] = raw[source:source + width * 4]
                    rgb = bytearray(width * height * 3)
                    rgb[0::3] = packed_bytes[2::4]
                    rgb[1::3] = packed_bytes[1::4]
                    rgb[2::3] = packed_bytes[0::4]
                    return RGBFrame(width, height, bytes(rgb))
                if np is None:
                    raise RuntimeError("unusual X image masks require NumPy")
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(meta.height, meta.bytes_per_line)
                pixels = rows[:, : meta.width * bytes_per_pixel].reshape(meta.height, meta.width, bytes_per_pixel)
                packed = pixels[:, :, :4].copy().view(
                    np.dtype("<u4" if meta.byte_order == 0 else ">u4")
                ).reshape(meta.height, meta.width)
                channels = []
                for mask in (meta.red_mask, meta.green_mask, meta.blue_mask):
                    shift = _mask_shift(int(mask))
                    bits = _mask_bits(int(mask))
                    maximum = (1 << bits) - 1 if bits else 1
                    channels.append((((packed & int(mask)) >> shift) * 255 // maximum).astype(np.uint8))
                array = np.stack(channels, axis=2)
                return RGBFrame(width, height, array.tobytes())
            finally:
                self.x11.XDestroyImage(image)

    def capture(self):
        frame = self._capture_rgb()
        if np is None:
            return frame
        return np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)

    def windows(self) -> list[WindowInfo]:
        root_return = ctypes.c_ulong()
        parent_return = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        with self._lock:
            ok = self.x11.XQueryTree(
                self.display, self.root, ctypes.byref(root_return), ctypes.byref(parent_return),
                ctypes.byref(children), ctypes.byref(count),
            )
            if not ok:
                return []
            try:
                ids = [int(children[index]) for index in range(count.value)]
            finally:
                if children:
                    self.x11.XFree(children)
            output = []
            for window_id in ids:
                title_ptr = ctypes.c_char_p()
                if self.x11.XFetchName(self.display, window_id, ctypes.byref(title_ptr)) and title_ptr.value:
                    title = title_ptr.value.decode("utf-8", errors="replace")
                    self.x11.XFree(title_ptr)
                    if title.strip():
                        output.append(WindowInfo(window_id, title.strip()))
            return output

    def tile(self) -> list[dict[str, Any]]:
        windows = self.windows()
        if not windows:
            return []
        width, height = self.size()
        count = len(windows)
        cell_width = max(320, width // count)
        layout = []
        with self._lock:
            for index, window in enumerate(windows):
                x = index * cell_width
                target_width = width - x if index == count - 1 else cell_width
                self.x11.XMoveResizeWindow(
                    self.display, window.window_id, x, 0,
                    max(1, target_width), max(1, height),
                )
                layout.append({
                    "window_id": window.window_id, "title": window.title,
                    "x": x, "y": 0, "width": target_width, "height": height,
                })
            self.x11.XFlush(self.display)
        return layout

    def focus(self, window_id: int) -> None:
        with self._lock:
            self.x11.XRaiseWindow(self.display, int(window_id))
            self.x11.XSetInputFocus(self.display, int(window_id), 1, 0)
            self.x11.XFlush(self.display)

    def focused_window_id(self) -> int | None:
        focused = ctypes.c_ulong()
        revert_to = ctypes.c_int()
        with self._lock:
            ok = self.x11.XGetInputFocus(
                self.display, ctypes.byref(focused), ctypes.byref(revert_to)
            )
        value = int(focused.value)
        return value if ok and value not in {0, self.root} else None

    def cycle_window(self, direction: int = 1) -> WindowInfo | None:
        """Raise and focus the next or previous titled top-level window."""
        windows = self.windows()
        if not windows:
            return None
        step = -1 if int(direction) < 0 else 1
        focused = self.focused_window_id()
        current = next(
            (index for index, item in enumerate(windows) if item.window_id == focused),
            None,
        )
        if current is None:
            target = windows[-1] if step < 0 else windows[0]
        else:
            target = windows[(current + step) % len(windows)]
        self.focus(target.window_id)
        return target

    def focus_tool(self, name: str) -> WindowInfo | None:
        """Focus a titled tool using stable human-facing names."""
        query = " ".join(str(name or "").casefold().split())
        if not query:
            return None
        aliases = {
            "paint": ("ina paint",),
            "drawing": ("ina paint",),
            "canvas": ("ina paint",),
            "daw": ("ina music studio",),
            "music": ("ina music studio",),
            "studio": ("ina music studio",),
            "music studio": ("ina music studio",),
        }
        candidates = aliases.get(query, (query,))
        windows = self.windows()
        target = next(
            (
                window for candidate in candidates for window in reversed(windows)
                if candidate == window.title.casefold() or candidate in window.title.casefold()
            ),
            None,
        )
        if target is not None:
            self.focus(target.window_id)
        return target

    def mouse_move(self, x: int, y: int) -> None:
        width, height = self.size()
        with self._lock:
            self.xtst.XTestFakeMotionEvent(
                self.display, -1, max(0, min(width - 1, int(x))),
                max(0, min(height - 1, int(y))), 0,
            )
            self.x11.XFlush(self.display)

    def mouse_button(self, button: int, pressed: bool) -> None:
        with self._lock:
            self.xtst.XTestFakeButtonEvent(self.display, max(1, int(button)), bool(pressed), 0)
            self.x11.XFlush(self.display)

    def key(self, keysym: str, pressed: bool) -> None:
        symbol = self.x11.XStringToKeysym(str(keysym).encode("utf-8"))
        keycode = self.x11.XKeysymToKeycode(self.display, symbol) if symbol else 0
        if not keycode:
            raise ValueError(f"unknown X keysym: {keysym}")
        with self._lock:
            self.xtst.XTestFakeKeyEvent(self.display, keycode, bool(pressed), 0)
            self.x11.XFlush(self.display)

    def type_text(self, text: str) -> None:
        keysyms = {
            "!": "exclam", '"': "quotedbl", "#": "numbersign", "$": "dollar",
            "%": "percent", "&": "ampersand", "'": "apostrophe", "(": "parenleft",
            ")": "parenright", "*": "asterisk", "+": "plus", ",": "comma",
            "-": "minus", ".": "period", "/": "slash", ":": "colon",
            ";": "semicolon", "<": "less", "=": "equal", ">": "greater",
            "?": "question", "@": "at", "[": "bracketleft", "\\": "backslash",
            "]": "bracketright", "^": "asciicircum", "_": "underscore",
            "`": "grave", "{": "braceleft", "|": "bar", "}": "braceright",
            "~": "asciitilde",
        }
        shifted = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+{}|:\"<>?")
        shift_down = False
        for char in str(text):
            name = "space" if char == " " else "Return" if char == "\n" else keysyms.get(char, char)
            needs_shift = char in shifted
            if needs_shift and not shift_down:
                self.key("Shift_L", True)
                shift_down = True
            elif shift_down and not needs_shift:
                self.key("Shift_L", False)
                shift_down = False
            self.key(name, True)
            self.key(name, False)
        if shift_down:
            self.key("Shift_L", False)

    def save_ppm(self, path: Path | str) -> Path:
        frame = self._capture_rgb()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(frame.ppm())
        return path
