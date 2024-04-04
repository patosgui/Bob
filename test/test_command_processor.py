import queue
from unittest import mock
from unittest.mock import patch


import command_processor
import audio_client
import audio_device
import ai_engine
import light_manager


@patch.object(audio_client.AudioClient, "reproduce", mock.Mock())
@patch.object(ai_engine.AIEngine, "predict")
@patch.object(light_manager.LightManager, "on")
def test_ai_trigger(mock_on, mock_predict):

    tts = mock.Mock()

    device = audio_device.Device(1, 48000)
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

    assert first_call_args[0] is 6 and first_call_args[1] is False
    assert second_call_args[0] is 8 and second_call_args[1] is False


@patch.object(audio_client.AudioClient, "reproduce", mock.Mock())
@patch.object(ai_engine.AIEngine, "predict")
@patch.object(light_manager.LightManager, "on")
def test_no_trigger_in_text(mock_on, mock_predict):
    tts = mock.Mock()
    device = audio_device.Device(1, 48000)
    client = audio_client.AudioClient(device=device)
    text_queue = queue.Queue()
    text_queue.put("Hey, Alice!")
    text_queue.put("Can you turn the kitchen light on?")
    mock_predict.return_value = "command: kitchen01 on"
    processor = command_processor.CommandProcessor(
        tts=tts, audio_client=client, text_queue=text_queue, trigger="Bob"
    )
    processor.process_once()
    mock_on.assert_not_called()


@patch.object(audio_client.AudioClient, "reproduce", mock.Mock())
@patch.object(ai_engine.AIEngine, "predict")
@patch.object(light_manager.LightManager, "on")
def test_invalid_command(mock_on, mock_predict):
    tts = mock.Mock()
    device = audio_device.Device(1, 48000)
    client = audio_client.AudioClient(device=device)
    text_queue = queue.Queue()
    text_queue.put("Hey, Bob!")
    text_queue.put("Can you turn the invalid light on?")
    mock_predict.return_value = "command: invalid01 on"
    processor = command_processor.CommandProcessor(
        tts=tts, audio_client=client, text_queue=text_queue, trigger="Bob"
    )
    processor.process_once()
    mock_on.assert_not_called()
