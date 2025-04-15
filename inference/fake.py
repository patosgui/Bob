import logging

from langchain_core.messages import AIMessage

from inference.ai_engine import AIEngine


class FakeModel(AIEngine):
    def analyze(self, sequence: str, max_length: int | None = None) -> None | AIMessage:
        logging.info(f"FakeModel: {sequence}")

        return None
