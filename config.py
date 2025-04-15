from dataclasses import dataclass
from typing import List


@dataclass
class GPT2Model:
    pass


@dataclass
class FakeModel:
    pass


@dataclass
class MistralModel:
    api_key: str


@dataclass
class HueBridge:
    ip: str


@dataclass
class Ollama:
    pass


@dataclass
class Config:
    conversation_model: GPT2Model | MistralModel | Ollama
    accessories: List[HueBridge]
