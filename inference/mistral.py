from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mistralai.chat_models import ChatMistralAI

import config
import light_manager
from inference.ai_engine import AIEngine

# Workaround llm tool calls and having a self
internal_lm = None


@tool
def turn_light_on(room: int) -> None:
    """Turns on the light in a room.

    Args:
        room: The room where the light is located. 0 for the office and 1 for the living room.
              The cozy lights are located in the living room.
    """
    assert internal_lm is not None

    if room == 0:
        internal_lm.on(6, True)
        internal_lm.on(8, True)
    elif room == 1:
        internal_lm.on(1, True)
        internal_lm.on(5, True)


@tool
def turn_light_off(room: int) -> None:
    """Turns off the light in a room.

    Args:
        room: The room where the light is located. 0 for the office and 1 for the living room.
              The cozy lights are located in the living room.
    """
    assert internal_lm is not None

    if room == 0:
        internal_lm.on(6, False)
        internal_lm.on(8, False)
    elif room == 1:
        internal_lm.on(1, False)
        internal_lm.on(5, False)


class MistralModel(AIEngine):
    def __init__(self, api_key: str, lm: light_manager.LightManager | None = None):
        global internal_lm
        internal_lm = lm

        llm = ChatMistralAI(model_name="mistral-large-latest", api_key=api_key)
        self.llm = llm.bind_tools([turn_light_on, turn_light_off])
        self.tool_map = {
            "turn_light_on": turn_light_on,
            "turn_light_off": turn_light_off,
        }

    def analyze(self, sequence: str, max_length: int | None = None) -> None | AIMessage:
        messages = [HumanMessage(content=sequence)]
        result = self.llm.invoke(messages)

        if not result.tool_calls:
            return result

        for tool_call in result.tool_calls:
            tool = self.tool_map[tool_call.get("name").lower()]
            tool.invoke(tool_call["args"])

        return None
