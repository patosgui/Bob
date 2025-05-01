import json
import logging

from openai import OpenAI
from openai.types.chat.chat_completion import Choice

# import light_manager
from inference.ai_engine import AIEngine

# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",  # required, but unused
# )
#
# response = client.chat.completions.create(
#     model="llama3.1",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The LA Dodgers won in 2020."},
#         {"role": "user", "content": "Where was it played?"},
#     ],
# )
# print(response.choices[0].message.content)


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a: The first integer number
      b: The second integer number

    Returns:
      int: The sum of the two numbers
    """
    return a + b


def turn_light_on(location: int) -> str:
    """
    Turns on the light in a room.

    Args:
        location: The room to the the light on. 0 for the office and 1 for the living room
    """
    # assert internal_lm is not None
    location = int(location)
    if location == 0:
        return "Light in the office is turned on"
        # internal_lm.on(6, True)
        # internal_lm.on(8, True)
    elif location == 1:
        return "Light in the living room is turned on"
        # internal_lm.on(1, True)
        # internal_lm.on(5, True)


class Ollama(AIEngine):
    client: OpenAI
    tools: list[dict]

    def __init__(self, lm=None):

        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "turn_light_on",
                    "description": "Turns on the light in a room.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "int",
                                "description": "The room to the the light on. 0 for the office and 1 for the living room",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ]

    def recurse_function_call(self, choice: Choice, messages: list[dict[str]]) -> str:
        if choice.finish_reason == "tool_calls":
            available_function = {
                "turn_light_on": turn_light_on,
            }

            # Append the answer from the model to which tool_call_id matches to.
            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                if tool_call.function.name in available_function:
                    dict_args = json.loads(tool_call.function.arguments)
                    logging.info(
                        f"Calling function: {tool_call.function.name} with args: {dict_args}"
                    )

                    try:
                        result = available_function[tool_call.function.name](
                            **dict_args
                        )
                    except Exception as e:
                        logging.error(
                            f"Error calling function: {tool_call.function.name} with args: {dict_args}"
                        )
                        logging.error(e)
                        return "Error calling function"

                    # Continue the conversation
                    print(f"Function call result: {result}")
                    assert isinstance(result, str)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Function not found",
                        }
                    )

            response = self.client.chat.completions.create(
                model="llama3.2",
                messages=messages,
                tools=self.tools,
            )

            return self.recurse_function_call(response.choices[0], messages)

        return choice.message.content

    def analyze(self, sequence: str, max_length: int | None = None) -> None:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful home assistant. You should give a direct answer to the user when the question is not related with home automation.",
            },
            {"role": "user", "content": f"{sequence}"},
        ]
        response = self.client.chat.completions.create(
            model="llama3.2",
            messages=messages,
            tools=self.tools,
        )
        return self.recurse_function_call(response.choices[0], messages)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )

    # response = ollama.client.chat.completions.create(
    #  model="llama3.1",
    #  messages=[
    #      {"role": "system", "content": "You are a helpful assistant."},
    #      {"role": "user", "content": "Who won the world series in 2020?"},
    #      {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    #      {"role": "user", "content": "Where was it played?"},
    #  ],
    # )
    # print(response)
    # print(response.choices[0].message.content)

    # response = ollama.client.chat.completions.create(
    #         model="llama3.1",
    #         messages=[
    #             {"role": "user", "content": "Can you turn the light on in the office?"}
    #         ],
    #         tools=ollama.tools,
    #     )

    # print("-----------------")

    # print(response)
    ol = Ollama(lm=None)
    print(ol.analyze(sequence="Can you turn the light on in the office?"))

    # print(completion.choices[0].message.content)
