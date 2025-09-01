import numpy as np

class EmotionSynth:
    """Minimal DDSP-style synthesizer mapping emotions to tones.

    The synthesizer provides a small set of continuous controls: pitch and
    loudness.  A linear layer maps 24 emotion sliders onto those controls.  It
    can render a simple sine-wave tone and perform a rudimentary training step
    to match a target fundamental frequency.  The goal is not high fidelity but
    to show that Ina can modulate low level audio features via internal
    control signals.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        # weights: 24 emotions -> [pitch_shift, loudness]
        self.weights = np.zeros((24, 2), dtype=float)

    # ------------------------------------------------------------------
    def emotion_to_controls(self, emotions: np.ndarray) -> tuple[float, float]:
        """Map a 24-dim emotion vector to (pitch_hz, loudness).

        Pitch is expressed in Hz around a 220 Hz base frequency.  Loudness is a
        linear gain in [0, 1].
        """
        emotions = np.asarray(emotions, dtype=float).reshape(24)
        pitch_shift, loud = emotions @ self.weights
        pitch = 220.0 + pitch_shift  # base A3
        loudness = np.clip(0.2 + loud, 0.0, 1.0)
        return pitch, loudness

    # ------------------------------------------------------------------
    def synthesize(self, emotions: np.ndarray, duration: float = 1.0) -> np.ndarray:
        """Render a tone for the given emotion vector."""
        pitch, amp = self.emotion_to_controls(emotions)
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        wave = amp * np.sin(2 * np.pi * pitch * t)
        return wave.astype(np.float32)

    # ------------------------------------------------------------------
    def estimate_pitch(self, audio: np.ndarray) -> float:
        """Estimate dominant frequency using an FFT peak."""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
        return float(freqs[int(np.argmax(fft))])

    # ------------------------------------------------------------------
    def train_pitch(self, emotions: np.ndarray, target_freq: float,
                    lr: float = 0.01, steps: int = 100) -> None:
        """Simple gradient descent on the pitch weight column.

        It updates the internal emotion->pitch mapping so that the generated
        tone's pitch moves closer to ``target_freq`` for the provided emotion
        vector.  This roughly corresponds to Stage‑1 vowel matching in the
        proposed learning loop.
        """
        emotions = np.asarray(emotions, dtype=float).reshape(24)
        for _ in range(steps):
            pitch, _ = self.emotion_to_controls(emotions)
            error = pitch - target_freq
            # gradient of pitch w.r.t weights is the emotion vector
            self.weights[:, 0] -= lr * error * emotions
