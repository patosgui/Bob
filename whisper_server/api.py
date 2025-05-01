import json
import logging
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.requests import Request

# Import the text_queue to communicate via test
# Workaround for passing arguments to FastAPI endpoints
import global_vars

router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

# WebSocket endpoint
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from C++ client
            data = await websocket.receive_text()
            logging.info(f"Message received from C++ client: {data}")

            # Parse JSON message
            try:
                json_data = json.loads(data)
                # Process the message (example)
                if "command" in json_data:
                    if json_data["command"] == "process":
                        msg = json_data.get("data", "No data")
                        result = {
                            "status": "success",
                            "result": f"Processed: {msg}",
                        }
                        global_vars.text_queue.put(msg)
                    else:
                        result = {"status": "error", "message": "Unknown command"}
                else:
                    result = {"status": "ack", "message": "Message received"}

                # Send response back to the C++ client
                # await manager.send_message(json.dumps(result), websocket)
            except json.JSONDecodeError:
                # If not JSON, just echo back
                await manager.send_message(f"Echo: {data}", websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logging.info("Client disconnected")


# Regular HTTP endpoint for testing
@router.websocket_route("/")
async def get():
    return {"message": "WebSocket server is running. Connect to /ws endpoint."}
