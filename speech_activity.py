#!/usr/bin/env python3
"""Lightweight speech activity detector with fan-noise tolerance."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

import numpy as np

try:  # optional dependency
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sd = None


@dataclass
class SpeechActivityConfig:
    sample_rate: int = 16000
    window_sec: float = 0.5
    min_interval: float = 0.5
    snr_threshold: float = 1.6
    band_ratio_threshold: float = 0.45
    min_rms: float = 0.004
    noise_decay: float = 0.96
    pre_emphasis: float = 0.97
    band_low_hz: float = 300.0
    band_high_hz: float = 3400.0

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "SpeechActivityConfig":
        if not isinstance(cfg, dict):
            return cls()
        return cls(
            sample_rate=int(cfg.get("sample_rate", cls.sample_rate)),
            window_sec=float(cfg.get("window_sec", cls.window_sec)),
            min_interval=float(cfg.get("min_interval", cls.min_interval)),
            snr_threshold=float(cfg.get("snr_threshold", cls.snr_threshold)),
            band_ratio_threshold=float(cfg.get("band_ratio_threshold", cls.band_ratio_threshold)),
            min_rms=float(cfg.get("min_rms", cls.min_rms)),
            noise_decay=float(cfg.get("noise_decay", cls.noise_decay)),
            pre_emphasis=float(cfg.get("pre_emphasis", cls.pre_emphasis)),
            band_low_hz=float(cfg.get("band_low_hz", cls.band_low_hz)),
            band_high_hz=float(cfg.get("band_high_hz", cls.band_high_hz)),
        )


@dataclass
class SpeechActivityResult:
    active: bool
    rms: float
    snr: float
    band_ratio: float
    timestamp: float


class SpeechActivityMonitor:
    def __init__(
        self,
        *,
        config: Optional[SpeechActivityConfig] = None,
        device: Optional[object] = None,
    ) -> None:
        self.config = config or SpeechActivityConfig()
        self.device = device
        self._stream = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: Deque[np.ndarray] = deque()
        self._queue_lock = threading.Lock()
        self._data_event = threading.Event()
        self._noise_floor: Optional[float] = None
        self._last_emit = 0.0
        self._last_active: Optional[bool] = None
        self._callback: Optional[Callable[[SpeechActivityResult], None]] = None

    @property
    def is_available(self) -> bool:
        return sd is not None

    def start(self, callback: Callable[[SpeechActivityResult], None]) -> bool:
        if sd is None:
            return False
        if self._stream is not None:
            return True
        self._callback = callback
        self._stop_event.clear()
        window_samples = max(1, int(self.config.sample_rate * self.config.window_sec))
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                device=self.device,
                blocksize=window_samples,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            self._stream = None
            return False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._data_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None

    def _audio_callback(self, indata, _frames, _time_info, _status) -> None:
        if self._stop_event.is_set():
            return
        chunk = np.array(indata[:, 0], dtype=np.float32, copy=True)
        with self._queue_lock:
            self._queue.append(chunk)
        self._data_event.set()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._data_event.wait(0.2):
                continue
            while True:
                with self._queue_lock:
                    if not self._queue:
                        self._data_event.clear()
                        break
                    chunk = self._queue.popleft()
                result = self._detect(chunk)
                if result is None:
                    continue
                now = result.timestamp
                if self._should_emit(result, now):
                    self._last_emit = now
                    self._last_active = result.active
                    if self._callback is not None:
                        try:
                            self._callback(result)
                        except Exception:
                            pass

    def _should_emit(self, result: SpeechActivityResult, now: float) -> bool:
        if self._last_active is None:
            return True
        if result.active != self._last_active:
            return True
        return (now - self._last_emit) >= self.config.min_interval

    def _detect(self, samples: np.ndarray) -> Optional[SpeechActivityResult]:
        if samples.size == 0:
            return None
        cfg = self.config
        data = samples.astype(np.float32, copy=False)
        if cfg.pre_emphasis:
            data = np.append(data[0], data[1:] - cfg.pre_emphasis * data[:-1])

        rms = float(np.sqrt(np.mean(np.square(data))))
        if self._noise_floor is None:
            self._noise_floor = rms

        window = np.hanning(len(data)).astype(np.float32)
        spectrum = np.fft.rfft(data * window)
        power = np.abs(spectrum) ** 2
        total_energy = float(np.sum(power) + 1e-9)
        freqs = np.fft.rfftfreq(len(data), d=1.0 / cfg.sample_rate)
        band_mask = (freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)
        band_energy = float(np.sum(power[band_mask]))
        band_ratio = band_energy / total_energy if total_energy > 0 else 0.0

        noise_floor = max(self._noise_floor or 0.0, 1e-6)
        snr = rms / noise_floor

        active = bool(
            rms >= cfg.min_rms
            and snr >= cfg.snr_threshold
            and band_ratio >= cfg.band_ratio_threshold
        )

        if not active:
            self._noise_floor = (cfg.noise_decay * noise_floor) + ((1.0 - cfg.noise_decay) * rms)

        return SpeechActivityResult(
            active=active,
            rms=rms,
            snr=snr,
            band_ratio=band_ratio,
            timestamp=time.time(),
        )
