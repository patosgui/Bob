# Python FastAPI WebSocket Server
# Requirements: pip install fastapi uvicorn websockets

import uvicorn
from fastapi import FastAPI

from whisper_server.api import router


def run_whisper_cpp(app):
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    print("Starting FastAPI WebSocket server on http://localhost:8000")
    print("WebSocket endpoint available at ws://localhost:8000/ws")
    app = FastAPI()
    run_whisper_cpp(app)
