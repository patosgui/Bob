from transformers import GPT2LMHeadModel, GPT2Tokenizer

from pathlib import Path

from gpt2.gpt2 import GPT2Model


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


class AIEngine:
    MODEL_PATH = Path(__file__).parent / "gpt2" / "results"

    def __init__(self):
        gpt2_model = GPT2Model(AIEngine.MODEL_PATH)
        self.model = gpt2_model.model
        self.tokenizer = gpt2_model.tokenizer

    def predict(self, sequence: str, max_length: int = 180):
        ids = self.tokenizer.encode(f"{sequence}", return_tensors="pt")
        final_outputs = self.model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )

        output = self.tokenizer.decode(
            final_outputs[0], skip_special_tokens=True
        )
        return output
