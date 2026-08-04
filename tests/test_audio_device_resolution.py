import pytest

from audio_device_resolution import resolve_audio_device


class FakeSoundDevice:
    def __init__(
        self,
        devices,
        *,
        default_input=-1,
        default_output=-1,
        rejected_output_devices=(),
    ):
        self.devices = list(devices)
        self.default_input = default_input
        self.default_output = default_output
        self.rejected_output_devices = set(rejected_output_devices)
        self.query_calls = []
        self.input_checks = []
        self.output_checks = []

    def query_devices(self, device=None, kind=None):
        self.query_calls.append((device, kind))
        if device is None and kind is None:
            return list(self.devices)
        if device is None:
            device = self.default_input if kind == "input" else self.default_output
        if isinstance(device, str):
            matches = [
                index
                for index, info in enumerate(self.devices)
                if info.get("name") == device
            ]
            if len(matches) != 1:
                raise ValueError(f"Unknown audio device: {device}")
            device = matches[0]
        if isinstance(device, bool) or not isinstance(device, int):
            raise ValueError(f"Invalid audio device: {device}")
        if device < 0 or device >= len(self.devices):
            raise ValueError(f"Invalid audio device index {device}")
        info = self.devices[device]
        if kind and int(info.get(f"max_{kind}_channels", 0) or 0) < 1:
            raise ValueError(f"Not an {kind} device: '{info.get('name')}'")
        return info

    def check_input_settings(self, **settings):
        self.input_checks.append(settings)

    def check_output_settings(self, **settings):
        self.output_checks.append(settings)
        device_index = settings.get("device", self.default_output)
        if device_index in self.rejected_output_devices:
            raise ValueError(f"Unsupported output settings for {device_index}")


def device(name, *, inputs=0, outputs=0):
    return {
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


def resolve_output(config, sounddevice):
    return resolve_audio_device(
        config,
        label="output_headset",
        role="output",
        index_keys=(
            "output_headset_index",
            "ouput_headset_index",
            "output_TV_index",
            "ouput_TV_index",
        ),
        name_keys=(
            "output_headset_name",
            "ouput_headset_name",
            "output_TV_name",
            "ouput_TV_name",
        ),
        sounddevice_module=sounddevice,
        sample_rate=44_100,
        channels=2,
    )


def test_default_override_wins_over_stale_and_input_only_indices():
    sounddevice = FakeSoundDevice(
        [
            device("System speakers", outputs=2),
            device("Unused"),
            device("Unused 2"),
            device("Unused 3"),
            device("Unused 4"),
            device("Unused 5"),
            device("Unused 6"),
            device("Unused 7"),
            device("Unused 8"),
            device("beyerdynamic", outputs=2),
            device("Unused 10"),
            device("HD-Audio Generic Analog Stereo", inputs=2),
        ],
        default_input=11,
        default_output=0,
    )
    config = {
        "audio_device_overrides": {"output_headset": "default"},
        "output_headset_name": "default",
        "output_headset_index": 11,
        "ouput_headset_name": "beyerdynamic",
        "ouput_headset_index": 9,
    }

    result = resolve_output(config, sounddevice)

    assert result.available is True
    assert result.device is None
    assert result.name == "System speakers"
    assert result.source == "system_default"
    assert result.warning is None
    assert sounddevice.query_calls == [(None, "output")]
    assert sounddevice.output_checks == [
        {"samplerate": 44_100, "channels": 2, "dtype": "float32"}
    ]


@pytest.mark.parametrize("configured_index", [1, 99])
def test_wrong_role_or_stale_index_falls_back_to_valid_default(configured_index):
    sounddevice = FakeSoundDevice(
        [
            device("Default output", outputs=2),
            device("Input only", inputs=2),
        ],
        default_input=1,
        default_output=0,
    )

    result = resolve_output(
        {"output_headset_index": configured_index},
        sounddevice,
    )

    assert result.available is True
    assert result.device is None
    assert result.name == "Default output"
    assert result.source == "system_default"
    assert "configured output device could not be used" in result.warning.lower()


def test_valid_legacy_index_can_recover_from_invalid_canonical_index():
    sounddevice = FakeSoundDevice(
        [
            device("Default output", outputs=2),
            device("Unused"),
            device("Legacy headset", outputs=2),
        ],
        default_output=0,
    )

    result = resolve_output(
        {
            "output_headset_name": "Missing canonical",
            "output_headset_index": 99,
            "ouput_headset_name": "Legacy headset",
            "ouput_headset_index": 2,
        },
        sounddevice,
    )

    assert result.available is True
    assert result.device == 2
    assert result.name == "Legacy headset"
    assert result.source == "configured_index"


def test_reordered_valid_index_is_recovered_by_unique_saved_name():
    sounddevice = FakeSoundDevice(
        [
            device("Default output", outputs=2),
            device("HDMI output", outputs=2),
            device("beyerdynamic DT 990", outputs=2),
        ],
        default_output=0,
    )

    result = resolve_output(
        {
            "output_headset_name": "beyerdynamic",
            "output_headset_index": 1,
        },
        sounddevice,
    )

    assert result.available is True
    assert result.device == 2
    assert result.source == "configured_name"
    assert "matched" in result.warning.lower()


def test_ambiguous_saved_name_uses_default_instead_of_guessing():
    sounddevice = FakeSoundDevice(
        [
            device("Default output", outputs=2),
            device("USB Audio front", outputs=2),
            device("USB Audio rear", outputs=2),
        ],
        default_output=0,
    )

    result = resolve_output(
        {"output_headset_name": "USB Audio", "output_headset_index": 1},
        sounddevice,
    )

    assert result.device is None
    assert result.name == "Default output"
    assert result.source == "system_default"


def test_nondefault_override_is_authoritative_and_boolean_override_is_ignored():
    sounddevice = FakeSoundDevice(
        [
            device("System default", outputs=2),
            device("Wrong output", outputs=2),
            device("Explicit headphones", outputs=2),
        ],
        default_output=0,
    )

    explicit = resolve_output(
        {
            "audio_device_overrides": {"output_headset": 2},
            "output_headset_name": "default",
            "output_headset_index": 1,
        },
        sounddevice,
    )
    boolean = resolve_output(
        {"audio_device_overrides": {"output_headset": True}},
        sounddevice,
    )

    assert explicit.device == 2
    assert explicit.name == "Explicit headphones"
    assert explicit.source == "override"
    assert boolean.device is None
    assert boolean.name == "System default"


def test_output_settings_rejection_falls_back_or_stays_offline():
    fallback_device = FakeSoundDevice(
        [
            device("System default", outputs=2),
            device("Configured output", outputs=2),
        ],
        default_output=0,
        rejected_output_devices={1},
    )
    unavailable_device = FakeSoundDevice(
        [device("Unsupported default", outputs=2)],
        default_output=0,
        rejected_output_devices={0},
    )

    fallback = resolve_output(
        {"output_headset_index": 1},
        fallback_device,
    )
    unavailable = resolve_output({}, unavailable_device)

    assert fallback.available is True
    assert fallback.device is None
    assert fallback.name == "System default"
    assert "could not be used" in fallback.warning
    assert fallback_device.output_checks == [
        {
            "samplerate": 44_100,
            "channels": 2,
            "dtype": "float32",
            "device": 1,
        },
        {"samplerate": 44_100, "channels": 2, "dtype": "float32"},
    ]
    assert unavailable.available is False
    assert unavailable.source == "unavailable"


def test_explicit_device_index_zero_remains_valid():
    sounddevice = FakeSoundDevice(
        [
            device("Index zero output", outputs=2),
            device("System default", outputs=2),
        ],
        default_output=1,
    )

    result = resolve_output(
        {"output_headset_name": "Index zero output", "output_headset_index": 0},
        sounddevice,
    )

    assert result.device == 0
    assert result.source == "configured_index"


def test_missing_sounddevice_or_default_keeps_offline_studio_available():

    missing_dependency = resolve_output({}, None)
    no_default = resolve_output(
        {"output_headset_index": 4},
        FakeSoundDevice([device("Input only", inputs=1)]),
    )

    assert missing_dependency.available is False
    assert missing_dependency.source == "unavailable"
    assert "offline render" in missing_dependency.warning
    assert no_default.available is False
    assert no_default.device is None
    assert "offline render" in no_default.warning


def test_unconfigured_output_uses_default_without_a_warning():
    sounddevice = FakeSoundDevice(
        [device("Default output", outputs=1)],
        default_output=0,
    )

    result = resolve_output({}, sounddevice)

    assert result.available is True
    assert result.device is None
    assert result.channels == 1
    assert result.warning is None


def test_input_resolution_uses_role_specific_default_and_settings_check():
    sounddevice = FakeSoundDevice(
        [
            device("Default output", outputs=2),
            device("Default microphone", inputs=1),
        ],
        default_input=1,
        default_output=0,
    )

    result = resolve_audio_device(
        {
            "audio_device_overrides": {"mic_headset": "default"},
            "mic_headset_index": 0,
        },
        label="mic_headset",
        role="input",
        index_keys=("mic_headset_index",),
        name_keys=("mic_headset_name",),
        sounddevice_module=sounddevice,
        sample_rate=44_100,
        channels=1,
    )

    assert result.available is True
    assert result.device is None
    assert result.name == "Default microphone"
    assert sounddevice.input_checks == [
        {"samplerate": 44_100, "channels": 1, "dtype": "float32"}
    ]
