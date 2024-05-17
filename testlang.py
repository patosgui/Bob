from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.tools import tool
import asyncio


@tool
def turn_light_on(room: int) -> None:
    """Turns on the light in a room.

    Args:
        room: The room where the light is located. 0 for the office and 1 for the living room.
    """
    print("HELLO WORLD")
    return


def turn_light_off(room: int) -> None:
    """Turns off the light in a room.

    Args:
        room: The room where the light is located. 0 for the office and 1 for the living room.
    """
    return


async def chat_op(llm, content: str):
    messages = [HumanMessage(content=content)]
    result = await llm.ainvoke(messages)
    for tool_call in result.tool_calls:
        tool = {"turn_light_on": turn_light_on, "turn_light_off": turn_light_off}[
            tool_call.get("name").lower()
        ]
        tool.invoke(tool_call["args"])


# If api_key is not passed, default behavior is to use the `MISTRAL_API_KEY` environment variable.
llm = ChatMistralAI(
    model_name="mistral-large-latest", api_key="SrEVCj9X37tOrrdrsdlZ3ED6mSTkqViO"
)
llm_with_tools = llm.bind_tools([turn_light_on])

loop = asyncio.get_event_loop()
loop.run_until_complete(chat_op(llm_with_tools, "Can you turn the office light on?"))
