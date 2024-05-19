from pathlib import Path

import yaml
from marshmallow import Schema, ValidationError, fields, post_load, validates_schema

from config import *


class ParsingError(Exception):
    pass


class HueBridgeSchema(Schema):
    ip = fields.IP()


class AccessoriesSchema(Schema):
    hue_bridge = fields.Nested(HueBridgeSchema)


class GPT2Schema(Schema):
    # Use like: gpt2: {}
    pass


class MistralSchema(Schema):
    api_key = fields.Str(required=True)


class ConversationalModelSchema(Schema):
    gpt2 = fields.Nested(GPT2Schema)
    mistral = fields.Nested(MistralSchema)

    @validates_schema
    def validate_numbers(self, data, **kwargs):
        if "gpt2" in data and "mistral" in data:
            raise ValidationError("Only one conversation model is allowed")


class ConfigSchema(Schema):
    conversation_model = fields.Nested(ConversationalModelSchema, required=True)
    accessories = fields.Nested(AccessoriesSchema)

    @post_load
    def make_user(self, data, **kwargs):
        yaml_model = data["conversation_model"]
        if "mistral" in yaml_model:
            model = MistralModel(yaml_model["mistral"]["api_key"])
        if "gpt2" in yaml_model:
            model = GPT2Model()
        assert model

        bridge = None
        if "hue_bridge" in data["accessories"]:
            bridge = HueBridge(str(data["accessories"]["hue_bridge"]["ip"]))

        return Config(model, [bridge])


def load_config(file: Path):
    with open(file, "r") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    try:
        schema = ConfigSchema()
        return schema.load(config)
    except ValidationError as e:
        raise ParsingError(e)
