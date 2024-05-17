import pyaudio
import logging

from dataclasses import dataclass


@dataclass
class Device:
    device_number: int
    sample_rate: int


def list_devices():
    """
    List all audio devices registered with pyaudio.

    This function lists all audio devices registered with the pyaudio library
    by getting the device count from PyAudio and iterating to get the device
    info of each one, logging the info.
    """
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        logging.info(device_info)


def get_device(device_name: str) -> Device:
    """
    Get a Device object representing an audio device by name.

    Searches for a device with the given name by iterating through all devices and comparing names.
    If found, returns a Device object with the device index and default sample rate.
    Otherwise raises an Exception.
    """
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
