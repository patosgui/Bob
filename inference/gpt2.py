import re
import light_manager
from inference.ai_engine import AIEngine

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path


from gpt2.gpt2 import GPT2Model


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


class GPT2LocalModel(AIEngine):
    MODEL_PATH = Path(__file__).parent.parent / "gpt2" / "results"

    def __init__(self, lm: light_manager.LightManager):
        gpt2_model = GPT2Model(GPT2LocalModel.MODEL_PATH)
        self.model = gpt2_model.model
        self.tokenizer = gpt2_model.tokenizer

        self.lm = lm

    def analyze(self, sequence: str, max_length: int | None = 180):
        output = self.predict_action(sequence, max_length)
        self.perform_action(output)
        return None

    def predict_action(self, sequence: str, max_length: int | None = 180):
        ids = self.tokenizer.encode(f"{sequence}", return_tensors="pt")
        final_outputs = self.model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )

        output = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        return output

    def perform_action(self, output):
        if re.search("command.*entrance01.*on", output):
            self.lm.on(7, True)

        if re.search("command.*entrance01.*off", output):
            self.lm.on(7, False)

        if re.search("command.*office01.*on", output):
            self.lm.on(6, True)
            self.lm.on(8, True)

        if re.search("command.*office01.*off", output):
            self.lm.on(6, False)
            self.lm.on(8, False)

        if re.search("command.*livingroom01.*off", output):
            self.lm.on(1, False)
            self.lm.on(5, False)

        if re.search("command.*livingroom01.*on", output):
            self.lm.on(1, True)
            self.lm.on(5, True)
