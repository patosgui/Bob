import re
import queue

import audio_client
import ai_engine
import light_manager

from abc import abstractmethod


class Debug:
    @abstractmethod
    def initializationOver(self):
        pass

    @abstractmethod
    def gotText(self, text: str):
        pass

    @abstractmethod
    def triggerAI(self):
        pass

    @abstractmethod
    def processingCommand(self, text: str):
        pass

    @abstractmethod
    def inferenceResult(self, result: str):
        pass


class CommandProcessor:
    def __init__(
        self,
        tts,
        audio_client: audio_client.AudioClient,
        text_queue: queue.Queue,
        ai_engine: ai_engine.AIEngine = ai_engine.AIEngine(),
        lm: light_manager.LightManager = light_manager.LightManager(),
        debug: Debug = Debug(),
        trigger: str = "Bob",
    ) -> None:
        self.tts = tts
        self.debug = debug
        self.trigger = trigger
        self.aie = ai_engine
        self.lm = lm
        self.audio_client = audio_client
        self.text_queue = text_queue

    def process_once(self):
        text = self.wait_for_new_data()
        self.debug.gotText(text)
        if self.trigger in text:
            try:
                if self.tts:
                    wav = self.tts.tts(text="Yes please.")
                    self.audio_client.reproduce(
                        wav, self.tts.synthesizer.output_sample_rate
                    )
                self.debug.triggerAI()
                # wait 10 second for a command
                cmd = self.wait_for_new_data(timeout=10)
                self.debug.processingCommand(cmd)
                self.process(cmd)
            except queue.Empty as e:
                # Ignore if command was not given
                pass

    def start(self):
        while True:
            self.process_once()

    def process(self, cmd):
        output = self.aie.predict(cmd, 180)
        self.debug.inferenceResult(output)
        # if "kitchen01" in output:
        #    await lm.set_light_on()

        if re.search("command.*entrance01.*on", output):
            self.lm.on(7, True)

        if re.search("command.*entrance01.*off", output):
            self.lm.on(7, False)

        if re.search("command.*office01.*on", output):
            self.lm.on(6, True)
            self.lm.on(8, True)

        if re.search("command.*office01.*off", output):
            self.lm.on(6, False)
            self.lm.on(8, False)

        if re.search("command.*livingroom01.*off", output):
            self.lm.on(1, False)
            self.lm.on(5, False)

        if re.search("command.*livingroom01.*on", output):
            self.lm.on(1, True)
            self.lm.on(5, True)

    def wait_for_new_data(self, timeout=None):
        data = self.text_queue.get(timeout=timeout)
        return data
