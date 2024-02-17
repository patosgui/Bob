import logging

from govee_api_laggat import Govee
from phue import Bridge
from dataclasses import dataclass
from typing import List
import govee_api_laggat


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
        govee_api_key: str = "72aa9b5d-2670-4d08-8b22-e8d0a48f7897",
        connect: bool = True,
    ):
        self.govee_api_key = govee_api_key
        self.bridge = Bridge("192.168.2.111")
        if connect:
            self.connect()

    def connect(self):
        # If the app is not registered and the button is not pressed, press the button and call connect()
        # (this only needs to be run a single time)
        self.bridge.connect()

    @property
    def status(self) -> List[Light]:
        lights = {}
        for light in self.bridge.lights:
            lights[light.light_id] = Light(light.name, light.light_id, light.on)
        return lights

    def on(self, light_id: int, value: bool):
        # TODO Add a map from the abstract id to the actual id. Only need to be done when supporting multiple APIs
        try:
            lights_by_id = self.bridge.get_light_objects(mode="id")
            lights_by_id[light_id].on = value
        except KeyError as e:
            raise LightNotFound("Light with id {id} could not be found!") from e

    async def set_light_on(self):
        async with Govee(self.govee_api_key) as govee:
            devices, err = await govee.get_devices()
            for dev in devices:
                # configure each device to turn on before seting brightness
                dev.before_set_brightness_turn_on = True
                dev.learned_set_brightness_max = 100
                # turn and set bright
                logging.info(f"Turning the light on (device: {dev})")
                await govee.set_brightness(dev, 254)

    async def set_hue_light_off(self):
        self.bridge.set_light([1, 2, 3], "on", False)
