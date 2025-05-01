import queue

import numpy as np

import audio_client
from inference import ai_engine


class Debug:
    def initializationOver(self):
        pass

    def gotText(self, text: str):
        pass

    def triggerAI(self):
        pass

    def processingCommand(self, text: str):
        pass

    def inferenceResult(self, result: str):
        pass


import logging


class CommandProcessor:
    def __init__(
        self,
        tts,
        audio_client: audio_client.AudioClient,
        ai_engine: ai_engine.AIEngine,
        text_queue: queue.Queue,
        debug: Debug = Debug(),
        trigger: str = "Bob",
    ) -> None:
        self.tts = tts
        self.debug = debug
        self.trigger = trigger
        self.aie = ai_engine
        self.audio_client = audio_client
        self.text_queue = text_queue

    def reproduce_audio(self, text : str):
        logging.info("--- Running TTS inference:  ---")
        logging.info(f"Text: {text}")
        buffer = []
        for i, (sr, chunk) in enumerate(
            self.tts.stream_tts_sync(text, options={"voice_id": "tara"})
        ):
            buffer.append(chunk)
        wav = np.concatenate(buffer, axis=1)
        logging.info("--- Finished TTS inference ---")
        # Check data type and range
        self.audio_client.reproduce(wav[0], 24000)
        return 

    def process_once(self):
        text = self.wait_for_new_data()
        self.debug.gotText(text)
        if self.trigger in text:
            try:
                if self.tts:
                    self.reproduce_audio("How can I help you?")
                self.debug.triggerAI()
                # wait 10 second for a command
                cmd = self.wait_for_new_data(timeout=10)
                self.debug.processingCommand(cmd)
                answer = self.aie.analyze(cmd)
                if answer and self.tts:
                    assert isinstance(answer, str)
                    self.reproduce_audio(answer)
            except queue.Empty:
                # Ignore if command was not given
                pass

    def start(self):
        while True:
            self.process_once()

    def wait_for_new_data(self, timeout=None):
        data = self.text_queue.get(timeout=timeout)
        return data
