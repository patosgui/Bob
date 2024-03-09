from abc import abstractmethod
import audio_capture
import ai_engine
from audio_capture import AudioChannel
import light_manager
import threading
import re

from queue import Queue, Empty
from whisper_server import TranscriptionServer
from whisper_live.vad import VoiceActivityDetection

# The text queue used to pass data from the TranscriptionServer for further
# processing
text_queue = Queue()


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

    async def start(self, audio_channel: AudioChannel):
        # Run the voice detection as a thread
        server = TranscriptionServer(
            vad_model=VoiceActivityDetection(),
            audio_channel=audio_channel,
            text_queue=text_queue,
        )
        thread = threading.Thread(target=server.recv_audio)
        thread.start()

        self.debug.initializationOver()

        while True:
            text = self.wait_for_new_data()
            self.debug.gotText(text)
            if self.trigger in text:
                try:
                    wav = self.tts.tts(text="Hi! How can I help you?")
                    audio_capture.reproduce_wav(wav,self.tts.synthesizer.output_sample_rate)
                    self.debug.triggerAI()
                    # wait 10 second for a command
                    cmd = self.wait_for_new_data(timeout=10)
                    self.debug.processingCommand(cmd)
                    self.process(cmd)
                except Empty as e:
                    # Ignore if command was not given
                    pass

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

    def wait_for_new_data(self, timeout=None):
        data = text_queue.get(timeout=timeout)
        return data[0]["text"]
