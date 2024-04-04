import numpy as np
import pyaudio
import audio_client
import logging
import wave


class AudioRecorder:
    def __init__(self, device: audio_client.Device, chunk=48000):
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
