import queue
from unittest import mock
from unittest.mock import patch

import audio_client
import audio_device
import command_processor


@patch.object(audio_client.AudioClient, "reproduce", mock.Mock())
def test_ai_trigger():

    tts = mock.Mock()

    device = audio_device.Device(1, 48000)
    client = audio_client.AudioClient(device=device)

    text_queue = queue.Queue()
    text_queue.put("Hey, Bob!")
    text_queue.put("Can you turn the office light off?")

    aie = mock.Mock()
    processor = command_processor.CommandProcessor(
        tts=tts,
        audio_client=client,
        ai_engine=aie,
        text_queue=text_queue,
        trigger="Bob",
    )

    processor.process_once()

    aie.analyze.assert_called_once_with("Can you turn the office light off?")


@patch.object(audio_client.AudioClient, "reproduce", mock.Mock())
def test_no_trigger_in_text():
    tts = mock.Mock()
    device = audio_device.Device(1, 48000)
    client = audio_client.AudioClient(device=device)
    text_queue = queue.Queue()
    text_queue.put("Hey, Alice!")
    text_queue.put("Can you turn the kitchen light on?")
    aie = mock.Mock()
    processor = command_processor.CommandProcessor(
        tts=tts,
        audio_client=client,
        ai_engine=aie,
        text_queue=text_queue,
        trigger="Bob",
    )
    processor.process_once()
    aie.predict.assert_not_called()
