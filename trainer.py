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

# Additional checkpoint-related arguments in TrainingArguments
training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=500,  # Save model every 500 steps
    save_total_limit=3,  # Keep only the last 3 checkpoints
    num_train_epochs=3,
    learning_rate=2e-4,
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
    resume_from_checkpoint="./qlora_output/checkpoint-500"
)
trainer.save_model("polish-joke-gpt-qlora")