from queue import Queue
from typing import Any

# The text queue that is use to pass the text from whisper to the automation
text_queue: Queue[Any] = Queue()
