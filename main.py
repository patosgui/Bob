#!/usr/bin/env python3

import argparse
import logging
import audio_client
import automation_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Home automation bot")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default="72aa9b5d-2670-4d08-8b22-e8d0a48f7897",
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

    if args.input_source == "microphone":
        audio_device = audio_client.get_device(device_name=args.device)
        automation_pipeline.start_pipeline(
            audio_device=audio_device, log=automation_pipeline.Logger(logging)
        )
    elif args.input_source == "wav":
        # elif args.input_source == 'wav':
        #    audio_device = audio_client.get_device(device_name=args.wav_file)
        automation_pipeline.start_pipeline(
            audio_device=args.wav_file, log=automation_pipeline.Logger(logging)
        )
