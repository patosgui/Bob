import light_manager
import unittest
from dataclasses import dataclass


@dataclass
class fake_phue_light:
    name: str
    light_id: id
    on: bool


def test_status():
    lm = light_manager.LightManager(connect=False)
    lm.bridge = unittest.mock
    lm.bridge.lights = [fake_phue_light("Test", 2, True)]

    assert len(lm.status) == 1
    assert lm.status[2].name == "Test"
    assert lm.status[2].light_id == 2
    assert lm.status[2].on == True
