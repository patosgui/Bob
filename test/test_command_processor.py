import queue
from unittest import mock
from unittest.mock import patch


import command_processor
import audio_client
import ai_engine
import light_manager


@patch.object(audio_client.AudioClient, "reproduce", mock.Mock())
@patch.object(ai_engine.AIEngine, "predict")
@patch.object(light_manager.LightManager, "on")
def test_ai_trigger(mock_on, mock_predict):

    tts = mock.Mock()

    device = audio_client.Device(1, 48000)
    client = audio_client.AudioClient(device=device)

    text_queue = queue.Queue()
    text_queue.put("Hey, Bob!")
    text_queue.put("Can you turn the office light off?")
    mock_predict.return_value = "command: office01 off"

    processor = command_processor.CommandProcessor(
        tts=tts, audio_client=client, text_queue=text_queue, trigger="Bob"
    )

    processor.process_once()

    print(mock_on.call_args_list)
    first_call_args = mock_on.call_args_list[0][0]
    second_call_args = mock_on.call_args_list[1][0]

    assert first_call_args[0] == 6 and first_call_args[1] == False
    assert second_call_args[0] == 8 and second_call_args[1] == False
