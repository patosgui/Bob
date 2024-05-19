import logging
from dataclasses import dataclass
from typing import Any

from phue import Bridge

import config

logger = logging.getLogger(__name__)


@dataclass
class Light:
    name: str
    light_id: int
    on: bool


class LightNotFound(Exception):
    pass


class LightManager:
    def __init__(
        self,
        ip: str | None = None,
        connect: bool = True,
    ):
        self.bridge = Bridge(ip)
        if connect:
            self.connect()

    def connect(self):
        # If the app is not registered and the button is not pressed, press the button and call connect()
        # (this only needs to be run a single time)
        self.bridge.connect()

    @property
    def status(self) -> dict[Any, Light]:
        lights = {}
        for light in self.bridge.lights:
            lights[light.light_id] = Light(light.name, light.light_id, light.on)
        return lights

    def on(self, light_id: int, value: bool):
        # TODO Add a map from the abstract id to the actual id. Only need to be done when supporting multiple APIs
        logging.info(f'Turning light {light_id} {"on" if value else "off"}')
        try:
            lights_by_id = self.bridge.get_light_objects(mode="id")
            lights_by_id[light_id].on = value
        except KeyError as e:
            raise LightNotFound("Light with id {id} could not be found!") from e

    async def set_hue_light_off(self):
        self.bridge.set_light([1, 2, 3], "on", False)
