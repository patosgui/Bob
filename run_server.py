#!/usr/bin/env python3

import logging

from whisper_server import TranscriptionServer

if __name__ == "__main__":
    logging.basicConfig(
        encoding="utf-8", level=logging.INFO, handlers=[logging.StreamHandler()]
    )
    server = TranscriptionServer()
    server.run("0.0.0.0", 9091)
