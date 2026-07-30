from movement_drive import calculate_movement_urge


def test_curiosity_and_boredom_can_create_wandering_urge():
    result = calculate_movement_urge(
        {
            "values": {
                "curiosity": 0.98,
                "novelty": -0.08,
                "intensity": 0.95,
                "attention": 0.95,
                "stress": 0.08,
                "threat": 0.05,
            }
        },
        boredom=1.0,
        energy=1.0,
        sleep_pressure=0.0,
    )

    assert result["level"] > 0.6
    assert result["drivers"]["novelty"] == 0.0


def test_fatigue_and_threat_inhibit_movement():
    alert = calculate_movement_urge(
        {"curiosity": 1.0, "intensity": 1.0, "attention": 1.0},
        boredom=1.0, energy=1.0, sleep_pressure=0.0,
    )
    inhibited = calculate_movement_urge(
        {
            "curiosity": 1.0, "intensity": 1.0, "attention": 1.0,
            "stress": 1.0, "threat": 1.0,
        },
        boredom=1.0, energy=0.0, sleep_pressure=1.0,
    )

    assert inhibited["level"] < alert["level"]
    assert inhibited["drivers"]["inhibition"] == 0.75
