from enum import Enum
import yaml  # mypy: ignore-errors
from dataclasses import dataclass
from pathlib import Path

bridge = None
keys = None
models = None


@dataclass
class Keys:
    MistralAPI: str | None


@dataclass
class HueBridge:
    ip: str


class ModelType(Enum):
    GPT2 = "gpt2"
    Mistral = "mistral"


@dataclass
class Models:
    ActionModel: ModelType


def load_config(file: Path):
    global bridge
    global keys
    global models

    # FIXME: Use a proper parse into dataclasses
    with open(file, "r") as stream:
        data_loaded = yaml.safe_load(stream)
        if HueBridge.__name__ in data_loaded:
            bridge = HueBridge(**data_loaded[HueBridge.__name__])
        if Keys.__name__ in data_loaded:
            keys = Keys(**data_loaded[Keys.__name__])
        if Models.__name__ in data_loaded:
            models_map = data_loaded[Models.__name__]
            action_model = models_map["ActionModel"].lower()
            models = Models(ActionModel=ModelType(action_model))

    assert bridge is not None
    assert keys is not None
    assert models is not None
