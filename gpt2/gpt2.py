from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

import datasets
import torch
import evaluate
import numpy as np
from torch.nn import functional as F

from typing import Optional
from pathlib import Path

# https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners/notebook


class GPT2Model:
    MODEL_NAME = "gpt2"

    def __init__(self, model_path: Optional[Path] = None):
        # gpt2: vocab_size=50257
        if model_path:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2Model.MODEL_NAME)
            # Explanation: https://medium.com/@mayvic/solving-the-issue-of-falcon-text-generation-never-stopping-e8f599eae8f0
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        assert self.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(
            model_path if model_path else GPT2Model.MODEL_NAME
        )

        with torch.no_grad():
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_dataset(self, data_dir):
        ds = datasets.load_dataset("text", data_dir=data_dir)
        test_train_ds = ds["train"].train_test_split(test_size=0.2)
        tokenized_test_train_ds = test_train_ds.map(
            lambda batch: self.tokenizer(
                batch["text"], truncation=True, padding=True
            ),
            batched=True,
        )
        return tokenized_test_train_ds

    def load_data_collator(self, mlm=False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=mlm
        )
        return data_collator

    def generate_text(self, sequence, max_length):
        ids = self.tokenizer.encode(f"{sequence}", return_tensors="pt")
        print(ids)
        final_outputs = self.model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        print(self.tokenizer.decode(final_outputs[0]))


metric = evaluate.load("accuracy")


def freeze_model_for_training(model):
    # From https://github.com/alexcpn/transformer_learn/blob/gpt-loss-learn/gpt2_train_model.py
    # # Freeze bottom 10 layers
    # #https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d
    # for parameter in model.parameters():
    #     parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        # Only un-freeze the last n transformer blocks
        if i >= 10:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def train(
    data_dir,
    gpt2_model: GPT2Model,
    output_dir,
    overwrite_output_dir,
    per_device_train_batch_size,
    num_train_epochs,
):

    test_train_dataset = gpt2_model.load_dataset(data_dir)

    print(test_train_dataset["test"][0]["text"])
    print(
        gpt2_model.tokenizer.decode(test_train_dataset["test"][0]["input_ids"])
    )
    for iden in test_train_dataset["test"][0]["input_ids"]:
        gpt2_model.tokenizer.decode(test_train_dataset["test"][0]["input_ids"])
        print(iden, sep=" ")
    data_collator = gpt2_model.load_data_collator()

    # This does not work while "add_special_tokens" is set to true while
    # loading the tokenizer
    # tokenizer.save_pretrained(output_dir)

    freeze_model_for_training(gpt2_model.model)

    # model.save_pretrained(output_dir)
    # print(model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
    )

    def compute_metrics(eval_pred):
        debug = False

        logits, labels = eval_pred

        torch_logits = torch.from_numpy(logits)
        # get probabilities using softmax from logit score and convert it to numpy array

        # probabilities_scores is a logit with (batch_size, sequence_size, vocab_size)
        # where vocab_size has been converted into a vector of probabilities
        probabilities_scores = F.softmax(torch_logits, dim=-1).numpy()

        # Get the prediction for each token in the input sequence for each entry in the batch
        predictions = np.argmax(probabilities_scores, axis=-1)

        if debug:
            for prediction, label in zip(predictions, labels):
                print("\nPrediction:")
                for pred in prediction:
                    print("\t" + gpt2_model.tokenizer.decode(pred), end="")
                print("\nLabel:")
                for l in label:
                    if l > 0:
                        print(
                            f"\t {gpt2_model.tokenizer.decode(l) if l > 0  else 'NEG' }",
                            end="",
                        )
            print("\n")
        # Flatten the predicitons/labels which are list of lists.
        # The accuracy metric require only lists as arguments. Therefore we merge
        # the results of all sentence in the batch into the same list
        #
        # Important: Remember that the first label is nothing since there was no input
        # therefore, we remove it. In practice, this "left shifts" all elements in the list.
        # After this operation, the elements in the "labels" lists are matching the
        # "next token prediction" elements in the "predictions" list, instead of
        # being one token behind.
        predictions = predictions.flatten()[:-1]
        labels = labels.flatten()[1:]
        return metric.compute(predictions=predictions, references=labels)

    # Calculate accuracy
    trainer = Trainer(
        model=gpt2_model.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=test_train_dataset["train"],
        eval_dataset=test_train_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    gpt2_model.tokenizer.save_pretrained(output_dir)
    trainer.save_model()
