import importlib.util
import os
import sys
import types
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


_STUBBED_MODULES = [
    "tkinter",
    "PIL",
    "PIL.Image",
    "PIL.ImageColor",
    "PIL.ImageDraw",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULES}


def _rgb(color):
    if isinstance(color, tuple):
        return tuple(color[:3])
    if color == "red":
        return (255, 0, 0)
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    if color == "#fff":
        return (255, 255, 255)
    raise ValueError(f"unknown fake color: {color}")


def _image_color(color):
    if isinstance(color, tuple):
        return tuple(color)
    return _rgb(color) + (255,)


class FakeImage:
    def __init__(self, mode, size, bg):
        self.mode = mode
        self.size = size
        self.bg = _image_color(bg)
        self.pixels = {}

    def putpixel(self, point, color):
        self.pixels[(int(point[0]), int(point[1]))] = _image_color(color)

    def getpixel(self, point):
        return self.pixels.get((int(point[0]), int(point[1])), self.bg)

    def save(self, path):
        with open(path, "wb") as handle:
            handle.write(b"fake-png")


class FakeDraw:
    def __init__(self, image):
        self.image = image
        self.lines = []
        self.ellipses = []

    def line(self, coords, fill, width):
        self.lines.append((coords, fill, width))
        x1, y1, x2, y2 = coords
        self.image.putpixel(((x1 + x2) / 2, (y1 + y2) / 2), fill)

    def ellipse(self, bounds, fill, outline=None):
        self.ellipses.append((bounds, fill, outline))
        x1, y1, x2, y2 = bounds
        self.image.putpixel(((x1 + x2) / 2, (y1 + y2) / 2), fill)


class FakeImageModule:
    @staticmethod
    def new(_mode, size, bg):
        return FakeImage(_mode, size, bg)


class FakeImageColorModule:
    @staticmethod
    def getrgb(color):
        return _rgb(color)


class FakeImageDrawModule:
    @staticmethod
    def Draw(image):
        return FakeDraw(image)


def _install_gui_stubs():
    tk_module = types.ModuleType("tkinter")
    tk_module.ROUND = "round"
    tk_module.HORIZONTAL = "horizontal"
    tk_module.Tk = object
    tk_module.Canvas = object
    tk_module.Label = object
    tk_module.Scale = object
    tk_module.Frame = object
    tk_module.Button = object
    tk_module.Checkbutton = object
    tk_module.IntVar = object
    tk_module.BooleanVar = object
    tk_module.colorchooser = types.SimpleNamespace(askcolor=lambda **_kwargs: None)
    tk_module.messagebox = types.SimpleNamespace(
        askyesno=lambda *_args, **_kwargs: False,
        askyesnocancel=lambda *_args, **_kwargs: None,
        showerror=lambda *_args, **_kwargs: None,
        showinfo=lambda *_args, **_kwargs: None,
        showwarning=lambda *_args, **_kwargs: None,
    )
    tk_module.simpledialog = types.SimpleNamespace(askstring=lambda *_args, **_kwargs: None)
    sys.modules["tkinter"] = tk_module

    pil_module = types.ModuleType("PIL")
    pil_module.Image = FakeImageModule
    pil_module.ImageColor = FakeImageColorModule
    pil_module.ImageDraw = FakeImageDrawModule
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = FakeImageModule
    sys.modules["PIL.ImageColor"] = FakeImageColorModule
    sys.modules["PIL.ImageDraw"] = FakeImageDrawModule


def _restore_modules():
    for name, module in _ORIGINAL_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_paint_window_under_test():
    _install_gui_stubs()
    try:
        spec = importlib.util.spec_from_file_location(
            "paint_window_under_test",
            os.path.join(ROOT, "paint_window.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules()


pw = _load_paint_window_under_test()


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeCanvas:
    def __init__(self):
        self.lines = []
        self.ovals = []
        self.deleted = []

    def create_line(self, *args, **kwargs):
        self.lines.append({"args": args, "kwargs": kwargs})

    def create_oval(self, *args, **kwargs):
        self.ovals.append({"args": args, "kwargs": kwargs})

    def delete(self, *args):
        self.deleted.append(args)


class FakeRoot:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _window_stub():
    window = pw.PaintWindow.__new__(pw.PaintWindow)
    window.root = FakeRoot()
    window.canvas = FakeCanvas()
    window.image = pw.Image.new("RGBA", (pw.CANVAS_WIDTH, pw.CANVAS_HEIGHT), pw.EMPTY_IMAGE_BG)
    window.draw = pw.ImageDraw.Draw(window.image)
    window.brush_size = FakeVar(6)
    window.is_eraser = FakeVar(False)
    window.active_color = pw.DEFAULT_COLOR
    window.dirty = False
    window._api_poll_active = True
    return window


class PaintWindowApiTests(unittest.TestCase):
    def test_normalise_paint_points_defaults_to_normalized_and_clamps(self):
        points = pw.normalise_paint_points(
            [[0.0, 0.0], {"x": 1.0, "y": 0.5}, [2.0, -1.0]],
            width=100,
            height=50,
        )

        self.assertEqual(points, [(0.0, 0.0), (100.0, 25.0), (100.0, 0.0)])

    def test_empty_workspace_uses_visible_backdrop_and_transparent_image(self):
        window = _window_stub()

        self.assertNotEqual(pw.DEFAULT_BG.lower(), "#ffffff")
        self.assertEqual(window.image.mode, "RGBA")
        self.assertEqual(window.image.getpixel((1, 1)), pw.EMPTY_IMAGE_BG)

        white = window._process_api_command(
            {
                "id": "white_line",
                "action": "stroke",
                "points": [[0.0, 0.0], [1.0, 1.0]],
                "color": "#ffffff",
            }
        )

        self.assertEqual(white["status"], "ok")
        self.assertEqual(window.canvas.lines[-1]["kwargs"]["fill"], "#ffffff")
        self.assertEqual(window.image.getpixel((450, 300)), (255, 255, 255, 255))

        eraser = window._process_api_command(
            {
                "id": "erase_line",
                "action": "stroke",
                "points": [[0.0, 0.0], [1.0, 1.0]],
                "eraser": True,
            }
        )

        self.assertEqual(eraser["status"], "ok")
        self.assertEqual(window.canvas.lines[-1]["kwargs"]["fill"], pw.DEFAULT_BG)
        self.assertEqual(window.image.getpixel((450, 300)), pw.EMPTY_IMAGE_BG)

    def test_api_stroke_draws_without_desktop_mouse(self):
        window = _window_stub()

        result = window._process_api_command(
            {
                "id": "stroke_1",
                "action": "stroke",
                "points": [[0.0, 0.0], [1.0, 1.0]],
                "color": "#d35400",
                "brush_size": 9,
            }
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["id"], "stroke_1")
        self.assertEqual(result["points"], 2)
        self.assertTrue(window.dirty)
        self.assertEqual(window.canvas.lines[0]["args"][:4], (0.0, 0.0, 900.0, 600.0))
        self.assertEqual(window.canvas.lines[0]["kwargs"]["fill"], "#d35400")
        self.assertEqual(window.image.getpixel((450, 300)), (211, 84, 0, 255))

    def test_string_false_eraser_does_not_turn_eraser_on(self):
        window = _window_stub()
        window.is_eraser.set(True)

        result = window._process_api_command(
            {
                "id": "dot_color",
                "action": "stroke",
                "points": [[0.5, 0.5]],
                "color": "#d35400",
                "eraser": "false",
            }
        )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["eraser"])
        self.assertEqual(window.canvas.ovals[0]["kwargs"]["fill"], "#d35400")

    def test_process_api_queue_clears_processed_commands_and_writes_status(self):
        window = _window_stub()
        store = {
            pw.PAINT_API_QUEUE_KEY: [
                {"id": "brush", "action": "set_brush", "brush_size": 14, "color": "red"},
                {"id": "dot", "action": "stroke", "points": [[10, 12]], "space": "pixels"},
            ]
        }
        original_get = pw.get_inastate
        original_update = pw.update_inastate

        try:
            pw.get_inastate = lambda key, default=None: store.get(key, default)
            pw.update_inastate = lambda key, value: store.__setitem__(key, value)

            window._process_api_queue()
        finally:
            pw.get_inastate = original_get
            pw.update_inastate = original_update

        self.assertEqual(store[pw.PAINT_API_QUEUE_KEY], [])
        self.assertEqual(store[pw.PAINT_API_LAST_RESULT_KEY]["processed"], 2)
        self.assertEqual(
            [item["id"] for item in store[pw.PAINT_API_LAST_RESULT_KEY]["results"]],
            ["brush", "dot"],
        )
        self.assertEqual(store[pw.PAINT_API_STATUS_KEY]["status"], "processed")
        self.assertEqual(window.brush_size.get(), 14)
        self.assertEqual(window.active_color, "#ff0000")
        self.assertTrue(window.canvas.ovals)

    def test_api_pattern_draws_generated_spiral_points(self):
        window = _window_stub()

        result = window._process_api_command(
            {
                "id": "spiral",
                "action": "pattern",
                "pattern": "spiral",
                "center": [0.5, 0.5],
                "radius": 0.2,
                "turns": 2,
                "steps": 24,
                "brush_size": 5,
            }
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pattern"], "spiral")
        self.assertEqual(result["strokes"], 1)
        self.assertEqual(result["points"], 25)
        self.assertEqual(len(window.canvas.lines), 24)
        self.assertTrue(window.dirty)

    def test_api_pattern_action_alias_draws_burst(self):
        window = _window_stub()

        result = window._process_api_command(
            {
                "id": "burst",
                "action": "burst",
                "center": [0.5, 0.5],
                "radius": 0.2,
                "rays": 6,
            }
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pattern"], "burst")
        self.assertEqual(result["strokes"], 6)
        self.assertEqual(result["points"], 12)
        self.assertEqual(len(window.canvas.lines), 6)

    def test_api_close_stops_polling_marks_closed_and_destroys_root(self):
        window = _window_stub()
        window.dirty = True
        store = {}
        original_update = pw.update_inastate

        try:
            pw.update_inastate = lambda key, value: store.__setitem__(key, value)

            result = window._process_api_command({"id": "done", "action": "close", "save": False})
        finally:
            pw.update_inastate = original_update

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["closed"])
        self.assertFalse(result["saved"])
        self.assertFalse(window._api_poll_active)
        self.assertTrue(window.root.destroyed)
        self.assertFalse(store["paint_window_open"])
        self.assertEqual(store[pw.PAINT_API_STATUS_KEY]["status"], "closed")


if __name__ == "__main__":
    unittest.main()
