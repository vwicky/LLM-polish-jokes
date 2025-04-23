from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import json

# Load jokes
with open("clean_polish_jokes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Hugging Face dataset
joke_texts = [{"text": entry["joke"]} for entry in data]
dataset = Dataset.from_list(joke_texts)

model_name = "flax-community/papuGaPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2-style

# Tokenize the dataset
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128  # Adjust as needed
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Save tokenized data if needed
tokenized_dataset.save_to_disk("tokenized_polish_jokes")

print("Tokenization complete. Ready for training.")
