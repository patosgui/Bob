"""
Implementation of an "AudioClient" that deals with everything that is audio
related. This includes both capturing data from a source and reproducing it
via e.g. a speaker.
"""

from dataclasses import dataclass
from queue import Queue
from typing import Optional

import numpy as np
import pyaudio
import json
import uuid
import logging
import librosa
import wave

from abc import ABC, abstractmethod


class AudioChannel(ABC):
    """
    An abstract (audio channel) used to pass data from the sender end e.g.
    a microphone to the receiving end e.g. a task or a function
    """

    @abstractmethod
    def recv():
        pass

    @abstractmethod
    def send():
        pass


class LocalAudioChannel(AudioChannel):
    """
    An implementation of AudioChannel that uses queues
    """

    def __init__(self):
        self.audio_queue = Queue()

    def send(self, message):
        # Put the received data into a queue
        self.audio_queue.put_nowait(message)

    def recv(self):
        return self.audio_queue.get()


@dataclass
class Device:
    device_number: int
    sample_rate: int


def list_devices():
    """
    List all devices registered with pyaudio
    """
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        logging.info(device_info)


def get_device(device_name: str) -> Optional[Device]:
    p = pyaudio.PyAudio()

    logging.info(f'Searching for device: "{device_name}"')
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_name in device_info["name"]:
            logging.info("Device found!")
            logging.info(device_info)
            return Device(i, int(device_info["defaultSampleRate"]))

    logging.info(list_devices())
    raise Exception(f"Device {device_name} not found.")


class AudioClient:
    def __init__(
        self,
        device: Device,
        is_multilingual: bool = False,
        lang: str = None,
        translate: bool = False,
    ):
        """
        Initializes a Client instance for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be established.
        When translate is True, the task will be set to "translate" instead of "transcribe".
        he audio recording starts immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            is_multilingual (bool, optional): Specifies if multilingual transcription is enabled. Default is False.
            lang (str, optional): The selected language for transcription when multilingual is disabled. Default is None.
            translate (bool, optional): Specifies if the task is translation. Default is False.
        """
        self.chunk = 8192
        self.format = pyaudio.paInt16
        self.channels = 1
        self.device = device
        self.rate = 16000
        self.recording = False
        self.multilingual = False
        self.language = None
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_recieved = None
        self.disconnect_if_no_response_for = 15
        self.multilingual = is_multilingual
        self.language = lang if is_multilingual else "en"
        if translate:
            self.task = "translate"

        self.timestamp_offset = 0.0
        self.audio_bytes = None
        self.p = pyaudio.PyAudio()

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0

    def on_open(self, audio_channel: AudioChannel):
        """
        Sends an initial configuration message to the server, including client UID, multilingual mode,
        language selection, and task type.
        """

        logging.info("[INFO]: Opened connection")
        audio_channel.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "multilingual": self.multilingual,
                    "language": self.language,
                    "task": self.task,
                }
            )
        )

    def record_once(self, stream, audio_channel):
        data = stream.read(self.chunk)

        audio_array = AudioClient.bytes_to_float_array(data)

        # Resample to target the 16kHz frequency for which the models
        # were trained
        resampled_data = librosa.resample(
            audio_array, orig_sr=self.device.sample_rate, target_sr=self.rate
        )

        audio_channel.send(resampled_data.tobytes())

    def record(self, audio_channel: AudioChannel):
        """
        Continuously capture data from the input device and stream it to the
        audio_channel
        """
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.device.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device.device_number,
        )

        logging.info("* recording")
        self.on_open(audio_channel=audio_channel)

        while True:
            self.record_once(stream=stream, audio_channel=audio_channel)

    def reproduce(self, wav, sample_rate):
        """
        Reprocuce a wav at the device's sample rate
        """

        # Use a local instance of pyaudio. Is pyaudio thread-safe?
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.device.sample_rate,
            input=False,
            frames_per_buffer=1024,
            output_device_index=self.device.device_number,
            output=True,
        )
        float32_array = np.array(wav, dtype=np.float32)

        resampled_data = librosa.resample(
            float32_array,
            orig_sr=sample_rate,
            target_sr=self.device.sample_rate,
        )

        float32_bytes = resampled_data.tobytes()
        stream.write(float32_bytes)

    def read_wav_file(self, audio_channel: AudioChannel, file_path):
        self.on_open(audio_channel=audio_channel)

        wav, sr = librosa.load(file_path)
        # librosa returns floats between -1 and 1. Multiply by 32767 to get int16 values
        wav = (wav * 32767).astype(np.int16)

        audio_array = AudioClient.bytes_to_float_array(wav)

        # Is this really required or does librosa make the downsampling?
        resampled_data = librosa.resample(
            audio_array, orig_sr=sr, target_sr=self.rate
        )
        self.chunk = 2048

        for i in range(0, len(resampled_data), self.chunk):
            chunk = resampled_data[i : i + self.chunk]
            audio_channel.send(chunk.tobytes())


class AudioRecorder:
    def __init__(self, device: Device, chunk=48000):
        self.device_number = device.device_number
        self.device_sample_rate = device.sample_rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.frames = []

    @staticmethod
    def bytes_to_float_array(data):
        return np.frombuffer(data, dtype=np.float32)

    def record(self, out_file="output_recording.wav"):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.device_sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device_number,
        )

        logging.info("[INFO]: Recording started")
        try:
            while True:
                data = stream.read(self.chunk)
                self.frames.append(data)

        except KeyboardInterrupt:
            logging.info("[INFO]: Recording stopped")
            stream.stop_stream()
            stream.close()
            self.p.terminate()

            # Concatenate all frames into a single audio array
            audio_data = b"".join(self.frames)
            # Convert to Flaot32
            audio_data = AudioRecorder.bytes_to_float_array(audio_data)

            wf = wave.open(out_file, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.device_sample_rate)
            wf.writeframes(b"".join(self.frames))
            wf.close()

            logging.info(f"[INFO]: Recording saved to {out_file}")
