from audio_capture import AudioChannel
from queue import Queue

import websocket
import threading
import logging

from websockets.sync.server import serve


class WebSocketAudioChannel(AudioChannel):

    def __init__(self, host: str, port: int, send: bool = True):
        """
        This class can only be configure for sending or receiving but not both.
        This is because it does not make sense to send and receive via a WebSocket in the same program
        """

        self.host = host
        self.port = port

        socket_url = f"ws://{host}:{port}"
        logging.info("Connection details: " + str(socket_url))

        self.audio_queue = Queue()

        if send:
            self.client_socket = websocket.WebSocketApp(
                socket_url,
                on_open=lambda ws: self.on_open(ws),
                on_message=lambda ws, message: self.on_message(ws, message),
                on_error=lambda ws, error: self.on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self.on_close(
                    ws, close_status_code, close_msg
                ),
            )

            # Run a thread forever to send and listen to answers
            self.ws_thread = threading.Thread(
                target=self.client_socket.run_forever
            )
            self.ws_thread.setDaemon(True)
            self.ws_thread.start()
        else:
            self.ws_thread = threading.Thread(target=self.recv_thread)
            self.ws_thread.setDaemon(True)
            self.ws_thread.start()

    def recv_thread(self):
        with serve(self.recv_callback, self.host, self.port) as server:
            server.serve_forever()

    def send_packet_to_server(self, message):
        """
        Send an audio packet to the server using WebSocket.

        Args:
            message (bytes): The audio data packet in bytes to be sent to the server.

        """
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            logging.error(e)

    def send(self, message):
        self.send_packet_to_server(message=message)

    def on_message(self, ws, message):
        """
        Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received message, including
        recording status, language detection, and server messages. If a disconnect message
        is received, it sets the recording status to False.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.
            message (str): The received message from the server.

        """
        pass

    def on_error(self, ws, error):
        logging.error(error)

    def on_close(self, ws, close_status_code, close_msg):
        logging.info(
            f"Websocket connection closed: {close_status_code}: {close_msg}"
        )

    def on_open(self, ws):
        """
        Callback function called when the WebSocket connection is successfully opened.

        Sends an initial configuration message to the server, including client UID, multilingual mode,
        language selection, and task type.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.

        """
        logging.info("[INFO]: Opened connection")
        ws.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "multilingual": self.multilingual,
                    "language": self.language,
                    "task": self.task,
                }
            )
        )

    def recv_callback(self, ws):
        while True:
            self.audio_queue.put_nowait(ws.recv())

    def recv(self):
        return self.audio_queue.get()

    def get_socket(self):
        return self.client_socket
