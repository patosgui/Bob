from unittest import mock
from unittest.mock import patch

from inference import gpt2


@patch.object(gpt2.GPT2LocalModel, "predict_action")
def test_valid_command_office_on(predict_action):
    predict_action.return_value = "command: office01 on"

    lm_mock = mock.Mock()
    gpt2_test = gpt2.GPT2LocalModel(lm_mock)
    gpt2_test.analyze(sequence="")

    first_call_args = lm_mock.on.call_args_list[0][0]
    second_call_args = lm_mock.on.call_args_list[1][0]
    assert first_call_args == (6, True)
    assert second_call_args == (8, True)


@patch.object(gpt2.GPT2LocalModel, "predict_action")
def test_valid_command_living_room_on(predict_action):
    predict_action.return_value = "command: livingroom01 on"

    lm_mock = mock.Mock()
    gpt2_test = gpt2.GPT2LocalModel(lm_mock)
    gpt2_test.analyze(sequence="")

    first_call_args = lm_mock.on.call_args_list[0][0]
    second_call_args = lm_mock.on.call_args_list[1][0]
    assert first_call_args == (1, True)
    assert second_call_args == (5, True)
