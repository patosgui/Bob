#!/usr/bin/env python3

# change sample rate for file for 1600 as expected by whisper
# import librosa
# y, s = librosa.load('StarWars60.wav', sr=16000)
# import soundfile as sf
# sf.write('stereo_file.wav', y, s)
# exit

#!/usr/bin/env python3

import logging
import audio_capture


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

device_number = audio_capture.search_device(
    device_name="Microphone (PowerConf S3)"
)

audio_channel = audio_capture.WebSocketAudioChannel(
            host="127.0.0.1", port=5676, send=True
        )

client = audio_capture.Client(
    device_number=device_number,
    audio_channel=audio_channel,
    is_multilingual=False,
    lang="en",
    translate=True,
)

client.record()

# client.play_file("C:\\Users\\lopes\\Downloads\\stereo_file.wav")
# client.play_file("C:\\Users\\lopes\\Downloads\\chunks\\hey_bob_office_light.wav")
# client.play_file("C:\\Users\\lopes\\Downloads\\chunks\\hey_bob_office_light.wav")
# client.play_file("C:\\Users\\lopes\\Downloads\\chunks\\hey_bob_kitchen_light.wav")
