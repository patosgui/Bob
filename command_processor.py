import queue

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

    def process_once(self):
        text = self.wait_for_new_data()
        self.debug.gotText(text)
        if self.trigger in text:
            try:
                if self.tts:
                    wav = self.tts.tts(
                        text="How can I help you?", language="en", speaker="male-en-2"
                    )
                    self.audio_client.reproduce(
                        wav, self.tts.synthesizer.output_sample_rate
                    )
                self.debug.triggerAI()
                # wait 10 second for a command
                cmd = self.wait_for_new_data(timeout=10)
                self.debug.processingCommand(cmd)
                answer = self.aie.analyze(cmd)
                if answer and self.tts:
                    wav = self.tts.tts(
                        text=answer.content, language="en", speaker="male-en-2"
                    )
                    self.audio_client.reproduce(
                        wav, self.tts.synthesizer.output_sample_rate
                    )
            except queue.Empty:
                # Ignore if command was not given
                pass

    def start(self):
        while True:
            self.process_once()

    def wait_for_new_data(self, timeout=None):
        data = self.text_queue.get(timeout=timeout)
        return data
