import audio_client
import pyaudio
import numpy as np

from unittest import mock
from unittest.mock import patch


def test_audio_client_initialization():
    """
    Test conversion from device with sample rate as 48kHz to 16kHz
    """

    device = audio_client.Device(1, 48000)
    client = audio_client.AudioClient(device=device)

    audio_channel = audio_client.LocalAudioChannel()

    # Create a chunk of zeros
    chunk = b"\x00\x00" * client.chunk
    mock_stream = mock.Mock()
    mock_stream.read.side_effect = [chunk]

    client.record_once(mock_stream, audio_channel)

    # Pass in: paInt16 chunk[8192] thus len(chunk) == 16304
    # chunk gets converted to paFloat32 thus len(chunk) == 8192
    # chunk gets cut to 2731 due to resampling thus len(chunk) = 2731
    # chunk gets converted from paFloat32 to bytes thus len(chunk) = 10924
    data = audio_channel.recv()

    assert client.chunk == 8192
    assert client.rate == 16000
    assert len(data) == 10924


@patch("pyaudio.PyAudio")
def test_reproduce(mock_pyaudio):
    """
    Test that reproducing audio at 48Khz from a wav sampled at 16kHz works
    """

    device = audio_client.Device(1, 48000)
    client = audio_client.AudioClient(device=device)

    mock_instance = mock_pyaudio.return_value

    stream = mock.Mock()
    mock_instance.open.return_value = stream

    wav = np.zeros(16000, dtype=np.float32)
    client.reproduce(wav, sample_rate=16000)

    # wav comes in with a 16000 chunk (or 1 second audio) of Float32 dtype
    # wav gets upsampled and goes from 16K elements to 48K elements
    # 48k Float32 get transformed into a byte array -> 192000
    assert len(stream.write.call_args[0][0]) == 192000
