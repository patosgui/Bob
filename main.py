#!/usr/bin/env python3

import argparse
import logging
import audio_device
import automation_pipeline
import config

from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Home automation bot")
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=Path(__file__).parent / "config.yml",
        help="The config file to use. By default it is expected to be in the same directory as this script",
    )

    subparsers = parser.add_subparsers(dest="input_source")
    microphone_parser = subparsers.add_parser("microphone")
    microphone_parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="PowerConf S3",
        help="The device used to capture and emit audio. The same device is used for both audio input and output",
    )
    wav_parser = subparsers.add_parser("wav")
    wav_parser.add_argument(
        "--wav-file", dest="wav_file", type=str, help="The file to play"
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

    # Set the global config
    config.load_config(args.config)

    if args.input_source == "microphone":
        audio_dev = audio_device.get_device(device_name=args.device)
        automation_pipeline.start_pipeline(
            audio_dev=audio_dev, log=automation_pipeline.Logger(logging)
        )
    elif args.input_source == "wav":
        automation_pipeline.start_pipeline(
            audio_dev=args.wav_file, log=automation_pipeline.Logger(logging)
        )
