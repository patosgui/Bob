#!/usr/bin/env python3

import argparse
import asyncio
import logging
import threading
from TTS.api import TTS

import server
import audio_capture


class Logger(server.Debug):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Home automation bot")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default="72aa9b5d-2670-4d08-8b22-e8d0a48f7897",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="PowerConf S3",
        help="The device used to capture and emit audio. The same device is used for both audio input and output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )

    device = audio_capture.search_device(device_name=args.device)
    audio_channel = audio_capture.LocalAudioChannel()
    client = audio_capture.Client(
        device=device,
        audio_channel=audio_channel,
        is_multilingual=False,
        lang="en",
        translate=True,
    )

    thread = threading.Thread(target=client.record)
    thread.start()

    log = Logger(logging)

    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to("cpu")
    cmd_processor = server.CommandProcessor(debug=log, tts=tts, device=device)
    res1 = asyncio.run(cmd_processor.start(audio_channel=audio_channel))
