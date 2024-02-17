#!/usr/bin/env python3

import gpt2

model = gpt2.GPT2Model()

gpt2.train(
    data_dir="dataset",
    gpt2_model=model,
    output_dir="results",
    overwrite_output_dir=False,
    per_device_train_batch_size=12,
    num_train_epochs=64,
)


model.generate_text(">Who was Sir Isaac Newton?", 40)
model.generate_text("Can you turn the kitchen light on?", 40)
model.generate_text("Turn the office light off", 40)
