from unittest import mock
from unittest.mock import patch

from inference import ollama


def test_turn_light_on():
    lm_mock = mock.Mock()
    ol_test = ollama.Ollama(lm_mock)

    ol_test.client = mock.Mock()

    print(ol_test.analyze(sequence="Can you turn the light on in the office?"))
