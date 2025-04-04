# Python FastAPI WebSocket Server
# Requirements: pip install fastapi uvicorn websockets

import json
from typing import List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

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
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from C++ client
            data = await websocket.receive_text()
            print(f"Message received from C++ client: {data}")

            # Parse JSON message
            try:
                json_data = json.loads(data)
                # Process the message (example)
                if "command" in json_data:
                    if json_data["command"] == "process":
                        result = {
                            "status": "success",
                            "result": f"Processed: {json_data.get('data', 'No data')}",
                        }
                    else:
                        result = {"status": "error", "message": "Unknown command"}
                else:
                    result = {"status": "ack", "message": "Message received"}

                # Send response back to the C++ client
                await manager.send_message(json.dumps(result), websocket)
            except json.JSONDecodeError:
                # If not JSON, just echo back
                await manager.send_message(f"Echo: {data}", websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")


# Regular HTTP endpoint for testing
@app.get("/")
async def get():
    return {"message": "WebSocket server is running. Connect to /ws endpoint."}


if __name__ == "__main__":
    print("Starting FastAPI WebSocket server on http://localhost:8000")
    print("WebSocket endpoint available at ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)
