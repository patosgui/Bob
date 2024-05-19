from dataclasses import dataclass
from typing import List


@dataclass
class GPT2Model:
    pass


@dataclass
class MistralModel:
    api_key: str


@dataclass
class HueBridge:
    ip: str


@dataclass
class Config:
    conversation_model: GPT2Model | MistralModel
    accessories: List[HueBridge]
