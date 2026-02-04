#!/usr/bin/env python3
"""PyQt main menu for the player client."""

from __future__ import annotations

from typing import Optional, Tuple

try:
    from PyQt5 import QtCore, QtWidgets
except Exception:  # pragma: no cover - optional dependency
    QtCore = None
    QtWidgets = None

try:
    from audio_settings_panel import AudioSettingsPanel, load_config, save_config
except Exception:  # pragma: no cover - optional dependency
    AudioSettingsPanel = None
    load_config = None
    save_config = None


def _get_player_name() -> str:
    if load_config is None:
        return "Player"
    cfg = load_config()
    name = cfg.get("player_name")
    if not name:
        return "Player"
    return str(name)


class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setModal(True)
        self._name_input = QtWidgets.QLineEdit()
        self._status = QtWidgets.QLabel("")
        self._audio_panel = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Player name"))
        self._name_input.setText(_get_player_name())
        name_row.addWidget(self._name_input)
        layout.addLayout(name_row)

        if AudioSettingsPanel is not None:
            self._audio_panel = AudioSettingsPanel()
            layout.addWidget(self._audio_panel)
        else:
            missing = QtWidgets.QLabel("Audio settings unavailable.")
            missing.setStyleSheet("color: #b0b0b0; font-size: 11px;")
            layout.addWidget(missing)

        self._status.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(self._status)

        buttons = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save")
        close_btn = QtWidgets.QPushButton("Close")
        save_btn.clicked.connect(self._save_name)
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(save_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _save_name(self) -> None:
        if load_config is None or save_config is None:
            self._status.setText("Config unavailable.")
            return
        name = self._name_input.text().strip() or "Player"
        cfg = load_config()
        cfg["player_name"] = name
        if save_config(cfg):
            self._status.setText("Saved player name.")
        else:
            self._status.setText("Failed to write config.json.")


class MainMenuDialog(QtWidgets.QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Inazuma Player Client")
        self.setModal(True)
        self.selection: Optional[str] = None
        self._name_label = QtWidgets.QLabel()
        self._build_ui()

    @property
    def player_name(self) -> str:
        return _get_player_name()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Inazuma Player Client")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        self._name_label.setText(f"Player: {_get_player_name()}")
        self._name_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        layout.addWidget(self._name_label)

        start_house = QtWidgets.QPushButton("Start House Viewer")
        start_house.clicked.connect(lambda: self._choose("house"))
        start_map = QtWidgets.QPushButton("Start Map Viewer")
        start_map.clicked.connect(lambda: self._choose("viewer"))
        layout.addWidget(start_house)
        layout.addWidget(start_map)

        console_label = QtWidgets.QLabel("Console only")
        console_label.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(console_label)

        console_row = QtWidgets.QHBoxLayout()
        headless_btn = QtWidgets.QPushButton("Headless")
        interactive_btn = QtWidgets.QPushButton("Interactive")
        headless_btn.clicked.connect(lambda: self._choose("headless"))
        interactive_btn.clicked.connect(lambda: self._choose("interactive"))
        console_row.addWidget(headless_btn)
        console_row.addWidget(interactive_btn)
        layout.addLayout(console_row)

        footer = QtWidgets.QHBoxLayout()
        options_btn = QtWidgets.QPushButton("Options")
        quit_btn = QtWidgets.QPushButton("Quit")
        options_btn.clicked.connect(self._open_options)
        quit_btn.clicked.connect(lambda: self._choose("quit"))
        footer.addWidget(options_btn)
        footer.addWidget(quit_btn)
        layout.addLayout(footer)

    def _choose(self, selection: str) -> None:
        self.selection = selection
        self.accept()

    def _open_options(self) -> None:
        dialog = OptionsDialog(self)
        dialog.exec_()
        self._name_label.setText(f"Player: {_get_player_name()}")


def run_player_menu() -> Tuple[str, str]:
    if QtWidgets is None:
        raise RuntimeError("PyQt5 unavailable")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = MainMenuDialog()
    dialog.exec_()
    selection = dialog.selection or "quit"
    return selection, dialog.player_name
