from transformers import AutoTokenizer
import json

# Load tokenizer
model_name = "szymonrucinski/Krakowiak-7B-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load cleaned jokes
with open("clean_all_jokes.json", "r", encoding="utf-8") as f:
    jokes = json.load(f)

# Tokenize all jokes and store tokenized info
tokenized_jokes = []
for joke in jokes:
    text = joke["joke"]
    tokens = tokenizer.encode(text, add_special_tokens=True)
    tokenized_jokes.append({
        "joke": text,
        "num_tokens": len(tokens),
        "tokens": tokens,
    })

# Save tokenized jokes
with open("tokenized_jokes_krakowiak.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_jokes, f, ensure_ascii=False, indent=2)

print(f"✅ Tokenized {len(tokenized_jokes)} jokes using {model_name}")
