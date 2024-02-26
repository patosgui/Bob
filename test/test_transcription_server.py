from whisper_server import TranscriptionServer

import time
import logging
from unittest import mock
from unittest.mock import patch
from queue import Queue
import threading

FAKE_CLIENT_DATA = [
    '{"uid": "53d0a251-808e-4531-861b-4d0c95dcbb30", "multilingual": true, "language": "en", "task": "translate"}',
    b"\x00\x00\x00\x00\x00\x00\x00\x00",
]


def test_timeout():
    text_queue = Queue()
    mock_vad_model = mock.Mock()
    ts = TranscriptionServer(vad_model=mock_vad_model, text_queue=text_queue)
    assert ts.max_connection_time == 3600 * 24  # 24 hours in seconds


@patch.object(TranscriptionServer, "client_provider", mock.Mock())
@patch.object(TranscriptionServer, "get_speech_probablity")
def test_recv_audio_no_voice(speech_prob, caplog):
    text_queue = Queue()
    mock_vad_model = mock.Mock()
    ts = TranscriptionServer(vad_model=mock_vad_model, text_queue=text_queue)

    mock_websocket = mock.Mock()
    # Pass some meta information to the server e.g. about which kind of language
    # is expected
    mock_websocket.recv.side_effect = FAKE_CLIENT_DATA
    speech_prob.return_value = 0.5

    with caplog.at_level(logging.DEBUG):
        thread = threading.Thread(target=ts.recv_audio, args=(mock_websocket,))
        thread.start()
        time.sleep(1)
        assert "New client connected" in caplog.text
        assert "Audio frame - size: 2 prob: 0.5" in caplog.text


@patch.object(TranscriptionServer, "client_provider", mock.Mock())
@patch.object(TranscriptionServer, "get_speech_probablity")
@patch.object(TranscriptionServer, "shouldTurnSpeechRecOff")
def test_recv_audio_voice(shoud_turn_off, speech_prob, caplog):
    text_queue = Queue()
    mock_vad_model = mock.Mock()
    ts = TranscriptionServer(vad_model=mock_vad_model, text_queue=text_queue)

    mock_websocket = mock.Mock()
    # Pass some meta information to the server e.g. about which kind of language
    # is expected
    mock_websocket.recv.side_effect = FAKE_CLIENT_DATA + [
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
    ]
    speech_prob.side_effect = [0.8, 0.3]
    # Do not go through the hassle of needing 0.5 seconds of data during testing
    shoud_turn_off.return_value = True

    with caplog.at_level(logging.DEBUG):
        thread = threading.Thread(target=ts.recv_audio, args=(mock_websocket,))
        thread.start()
        time.sleep(1)
        assert "New client connected" in caplog.text
        assert "Audio frame - size: 2 prob: 0.8" in caplog.text
        assert "Turning speech recognition on!" in caplog.text
        assert "Audio frame - size: 2 prob: 0.3" in caplog.text
        assert "Turning speech recognition off!" in caplog.text
