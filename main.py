#!/usr/bin/env python3

import argparse
import asyncio
import logging

import server


class Logger(server.Debug):
    def __init__(self, log):
        self.log = log

    def initializationOver(self):
        logging.info("** Ready **")

    def gotText(self, text: str):
        logging.info(f"Rcvd text: {text}")

    def triggerAI(self):
        logging.info(f"Listening for command!")

    def processingCommand(self, text: str):
        self.gotText(text)

    def inferenceResult(self, result: str):
        logging.info(f"** Inference result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="govee_api_laggat examples")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default="72aa9b5d-2670-4d08-8b22-e8d0a48f7897",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    log = Logger(logging)
    cmd_processor = server.CommandProcessor(debug=log)
    res1 = asyncio.run(cmd_processor.start())
