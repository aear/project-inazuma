import numpy as np
from motor_layer import EmotionSynth


def test_synthesize_length_and_range():
    synth = EmotionSynth(sample_rate=8000)
    emotions = np.zeros(24)
    wave = synth.synthesize(emotions, duration=0.5)
    assert wave.shape[0] == 4000
    assert np.max(np.abs(wave)) <= 1.0


def test_training_reduces_pitch_error():
    synth = EmotionSynth()
    emotions = np.zeros(24)
    emotions[0] = 1.0  # activate one slider
    target = 330.0
    before, _ = synth.emotion_to_controls(emotions)
    synth.train_pitch(emotions, target_freq=target, lr=0.05, steps=200)
    after, _ = synth.emotion_to_controls(emotions)
    assert abs(after - target) < abs(before - target)
