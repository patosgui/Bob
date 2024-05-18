import logging
import sys
import threading
import time
from queue import Queue
from typing import Any

from TTS.api import TTS
from whisper_live.vad import VoiceActivityDetection

import audio_client
import audio_device
import command_processor
import config
import light_manager
from inference import ai_engine, gpt2, mistral

# Based on whisper_live module
from transcription_server import TranscriptionServer


class Logger(command_processor.Debug):
    # FIXME: There shouldn't be a mighty log. Decompose it!

    def __init__(self, log):
        self.log = log

    def initializationOver(self):
        logging.info("** Ready **")

    def gotText(self, text: str):
        logging.info(f"Rcvd text: {text}")

    def triggerAI(self):
        logging.info("Listening for command!")

    def processingCommand(self, text: str):
        self.gotText(text)

    def inferenceResult(self, result: str):
        logging.info(f"** Inference result: {result}")


def start_pipeline(audio_dev: audio_device.Device | str, log: Logger):

    audio_channel = audio_client.LocalAudioChannel()

    # Prepare the audio capture thread
    client = None
    audio_capture_thread = None
    if isinstance(audio_dev, audio_device.Device):
        client = audio_client.AudioClient(
            device=audio_dev, is_multilingual=False, lang="en", translate=True
        )

        audio_capture_thread = threading.Thread(
            target=client.record, args=(audio_channel,), daemon=True
        )
    else:
        client = audio_client.AudioClient(
            device=None, is_multilingual=False, lang="en", translate=True
        )

        audio_capture_thread = threading.Thread(
            target=client.read_wav_file, args=(audio_channel, audio_dev)
        )

    # The text queue used to pass data from the TranscriptionServer for further
    # processing
    text_queue: Queue[Any] = Queue()

    # Run the voice detection as a thread
    server = TranscriptionServer(
        vad_model=VoiceActivityDetection(),
        audio_channel=audio_channel,
        text_queue=text_queue,
    )
    audio_receive_thread = threading.Thread(target=server.recv_audio, daemon=True)

    # Prepare the command processor
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cpu")

    aie: ai_engine.AIEngine | None = None

    assert config.models
    if config.models.ActionModel == config.ModelType.Mistral:
        aie = mistral.MistralModel()
    elif config.models.ActionModel == config.ModelType.GPT2:
        aie = gpt2.GPT2LocalModel(light_manager.LightManager())
    assert ai_engine

    cmd_process = command_processor.CommandProcessor(
        tts=tts,
        audio_client=client,
        text_queue=text_queue,
        ai_engine=aie,
        debug=log,
        trigger="Bob",
    )
    cmd_process_thread = threading.Thread(target=cmd_process.start, daemon=True)

    log.initializationOver()

    threads = [audio_capture_thread, audio_receive_thread, cmd_process_thread]

    try:
        for t in threads:
            t.start()
        while True:
            # Wake up every 0.2 to catch Ctrl+C and terminate gracefully
            time.sleep(0.2)
    except KeyboardInterrupt:
        sys.exit(1)
