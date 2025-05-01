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

    function_call = Function(arguments='{"location":"0"}', name="turn_light_on")
    # Create the ChatCompletionMessageToolCall object
    tool_call = ChatCompletionMessageToolCall(
        id="call_xip8pu1b", function=function_call, type="function", index=0
    )

    # Create the ChatCompletionMessage object
    message = ChatCompletionMessage(
        content="", role="assistant", tool_calls=[tool_call]
    )

    # Create the Choice object
    choice = Choice(finish_reason="tool_calls", index=0, message=message)

    second_choice = Choice(
        finish_reason="stop",
        index=0,
        logprobs=None,
        message=ChatCompletionMessage(
            content="How can I assist you further with your home automation needs?",
            refusal=None,
            role="assistant",
            annotations=None,
            audio=None,
            function_call=None,
            tool_calls=None,
        ),
    )

    mock_obj = mock.MagicMock()
    mock_obj.choices.__getitem__.return_value = choice

    mock_obj_2 = mock.MagicMock()
    mock_obj_2.choices.__getitem__.return_value = second_choice
    ol_test.client.chat.completions.create.side_effect = [mock_obj, mock_obj_2]

    turn_light_on_mock.return_value = "Light turned on"

    print(ol_test.analyze(sequence="Can you turn the light on in the office?"))

    assert turn_light_on_mock.called
