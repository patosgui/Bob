import audio_client
import threading
import command_processor
import logging
import sys
import time

from queue import Queue

from TTS.api import TTS
from whisper_live.vad import VoiceActivityDetection

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


def start_pipeline(audio_device: audio_client.Device, log: Logger):

    # Prepare the audio capture thread
    client = audio_client.AudioClient(
        device=audio_device, is_multilingual=False, lang="en", translate=True
    )

    audio_channel = audio_client.LocalAudioChannel()
    audio_capture_thread = threading.Thread(
        target=client.record, args=(audio_channel,), daemon=True
    )

    # The text queue used to pass data from the TranscriptionServer for further
    # processing
    text_queue = Queue()

    # Run the voice detection as a thread
    server = TranscriptionServer(
        vad_model=VoiceActivityDetection(),
        audio_channel=audio_channel,
        text_queue=text_queue,
    )
    audio_receive_thread = threading.Thread(
        target=server.recv_audio, daemon=True
    )

    # Prepare the command processor
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to("cpu")
    cmd_process = command_processor.CommandProcessor(
        tts=tts,
        audio_client=client,
        text_queue=text_queue,
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
    except KeyboardInterrupt as e:
        sys.exit(e)
