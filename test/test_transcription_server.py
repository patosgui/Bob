from whisper_server import TranscriptionServer

import time
from unittest import mock
from unittest.mock import patch
from queue import Queue
import threading
from whisper_live.vad import VoiceActivityDetection


def fake_bar(*args, **kwargs):
    return True


def fake_me(*args, **kwargs):
    return 0.5


@patch.object(TranscriptionServer, "client_provider", fake_bar)
@patch.object(TranscriptionServer, "get_speech_probablity")
def test_recv_audio(speech_prob):
    text_queue = Queue()
    mock_vad_model = mock.Mock()
    ts = TranscriptionServer(vad_model=mock_vad_model, text_queue=text_queue)

    mock_websocket = mock.Mock()
    # Pass some meta information to the server e.g. about which kind of language
    # is expected
    mock_websocket.recv.side_effect = [
        '{"uid": "53d0a251-808e-4531-861b-4d0c95dcbb30", "multilingual": true, "language": "en", "task": "translate"}',
        b"\x00\x00\x00\x00\x00\x00\x00\x00",
    ]
    speech_prob.return_value = 0.5

    thread = threading.Thread(target=ts.recv_audio, args=(mock_websocket,))
    thread.start()
    time.sleep(1)
    exit(1)
