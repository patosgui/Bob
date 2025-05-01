import logging
import sys
import threading
import time

from fastapi import FastAPI
from orpheus_cpp import OrpheusCpp

import audio_client
import audio_device
import command_processor
import config
import global_vars
import light_manager
from inference import ai_engine, ollama, mistral

# from whisper_live.vad import VoiceActivityDetection
from whisper_server.receiver import run_whisper_cpp

# Based on whisper_live module
# from transcription_server import TranscriptionServer


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


def start_pipeline(
    audio_dev: audio_device.Device | str, cfg: config.Config, log: Logger
):

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

    # # Run the voice detection as a thread
    # server = TranscriptionServer(
    #     vad_model=VoiceActivityDetection(),
    #     audio_channel=audio_channel,
    #     text_queue=text_queue,
    # )
    # audio_receive_thread = threading.Thread(target=server.recv_audio, daemon=True)

    # Prepare the command processor
    tts = OrpheusCpp(verbose=False)

    # Prepare the light manager
    lm: light_manager.LightManager | None = None
    logging.info(cfg.accessories)
    for accessory in cfg.accessories:
        if isinstance(accessory, config.HueBridge):
            lm = light_manager.LightManager()
    assert lm

    print(cfg)
    aie: ai_engine.AIEngine | None = None
    if isinstance(cfg.conversation_model, config.MistralModel):
        aie = mistral.MistralModel(api_key=cfg.conversation_model.api_key, lm=lm)
    elif isinstance(cfg.conversation_model, config.Ollama):
        aie = ollama.Ollama(lm=lm)

    assert ai_engine

    cmd_process = command_processor.CommandProcessor(
        tts=tts,
        audio_client=client,
        text_queue=global_vars.text_queue,
        ai_engine=aie,
        debug=log,
        trigger="Bob",
    )
    cmd_process_thread = threading.Thread(target=cmd_process.start, daemon=True)

    log.initializationOver()

    threads = [cmd_process_thread]

    try:
        for t in threads:
            t.start()

        # uvicorn needs to run in the main thread. Use asyncio for parallelism
        # app = FastAPI(lifespan=lambda app : lifespan(app, text_queue=text_queue))
        app = FastAPI()
        run_whisper_cpp(app)

        while True:
            # Wake up every 0.2 to catch Ctrl+C and terminate gracefully
            time.sleep(0.01)
    except KeyboardInterrupt:
        sys.exit(1)
