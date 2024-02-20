import light_manager
import unittest
from dataclasses import dataclass


@dataclass
class fake_phue_light:
    name: str
    light_id: id
    on: bool


def test_status():
    pass
