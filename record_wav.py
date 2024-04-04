import audio_recorder
import audio_client
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    force=True,
)

audio_device = audio_client.get_device(device_name="PowerConf S3")
audio_recorder.AudioRecorder(device=audio_device).record()
