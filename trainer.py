from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from trl import SFTTrainer

# Load tokenizer and model
model_name = "flax-community/papuGaPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map={"": 0},  # Explicitly load on GPU 0
)

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)
model = model.to("cuda")

# Add LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # GPT2-specific
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model = model.to("cuda")
print(f"Model moved to device: {next(model.parameters()).device}")

# Load tokenized dataset
dataset = load_from_disk("tokenized_polish_jokes")

from transformers import TrainerCallback
import random

class JokeLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt_list, log_every=500, max_new_tokens=40):
        self.tokenizer = tokenizer
        self.prompt_list = prompt_list
        self.log_every = log_every
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0 and state.global_step != 0:
            model = kwargs["model"]
            prompt = random.choice(self.prompt_list)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0
                )
            print(f"\n--- Sample joke @ step {state.global_step} ---")
            print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            print("-" * 50)


# Additional checkpoint-related arguments in TrainingArguments
training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
    save_steps=500,  # Save model every 500 steps
    save_total_limit=5,  # Keep only the last 3 checkpoints
    num_train_epochs=18,
    learning_rate=7e-5,
    fp16=True,
    report_to="none",
    # Ensure that checkpoints are saved as frequently as needed
    save_strategy="steps",  # Save after every 'save_steps'
    resume_from_checkpoint=True  # Automatically resumes from the last checkpoint
)


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define sample prompts (Polish joke starters)
sample_prompts = [
    "Przychodzi baba do lekarza i mówi",
    "Dlaczego blondynka weszła do sklepu",
    "Jasiu pyta nauczycielkę",
    "Facet wchodzi do baru i widzi"
]

# Add the callback to your SFTTrainer
callbacks = [JokeLoggerCallback(tokenizer, sample_prompts, log_every=150)]

# SFTTrainer simplifies LoRA + Trainer
trainer = SFTTrainer(
    model=model,
    #tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
print("Trainer device:", training_args.device)

import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

trainer.train(
    resume_from_checkpoint="./qlora_output/checkpoint-2200"
)
trainer.save_model("polish-joke-gpt-qlora")