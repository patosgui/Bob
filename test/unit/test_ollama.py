from unittest import mock
from unittest.mock import patch

from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from inference import ollama


@mock.patch("inference.ollama.turn_light_on")
def test_turn_light_on(turn_light_on_mock):
    lm_mock = mock.Mock()
    ol_test = ollama.Ollama(lm_mock)

    ol_test.client = mock.Mock()

    # Create the Choice for function call
    choice = Choice(
        finish_reason="tool_calls",
        index=0,
        message=ChatCompletionMessage(
            content="",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_xip8pu1b",
                    function=Function(
                        arguments='{"location":"0"}', name="turn_light_on"
                    ),
                    type="function",
                    index=0,
                )
            ],
        ),
    )

    # Create the Choice to stop the recursion and the interaction
    second_choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(
            content="How can I assist you further with your home automation needs?",
            role="assistant",
        ),
    )

    mock_obj_first = mock.MagicMock()
    mock_obj_first.choices.__getitem__.return_value = choice

    mock_obj_second = mock.MagicMock()
    mock_obj_second.choices.__getitem__.return_value = second_choice
    ol_test.client.chat.completions.create.side_effect = [
        mock_obj_first,
        mock_obj_second,
    ]

    turn_light_on_mock.return_value = "Light turned on"

    ol_test.analyze(sequence="Can you turn the light on in the office?")

    assert turn_light_on_mock.called
