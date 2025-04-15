from openai import OpenAI

import light_manager
from inference.ai_engine import AIEngine

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ],
)
print(response.choices[0].message.content)


class Ollama(AIEngine):
    def __init__(self, api_key: str, lm: light_manager.LightManager | None = None):

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "turn_light_on",
                    "description": "Turns on the light in a room.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
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

        completion = client.chat.completions.create(
            model="llama3.1",
            messages=[
                {"role": "user", "content": "Can you turn the light on in the office?"}
            ],
            tools=tools,
        )

    def analyze(self, sequence: str, max_length: int | None = None) -> None | AIMessage:
        messages = [HumanMessage(content=sequence)]
        result = self.llm.invoke(messages)

        if not result.tool_calls:
            return result

        for tool_call in result.tool_calls:
            tool = self.tool_map[tool_call.get("name").lower()]
            tool.invoke(tool_call["args"])

        return None
