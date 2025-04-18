import os
import json

def load_all_jokes_from_dir(dir_path):
    all_jokes = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    jokes = json.load(f)
                    all_jokes.extend(jokes)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipped invalid JSON: {filename}")
    return all_jokes

def preprocess_jokes(jokes, min_length=20, max_length=1000):
    seen = set()
    cleaned = []

    for entry in jokes:
        if isinstance(entry, dict) and "joke" in entry:
            joke = entry["joke"].strip()
            if min_length < len(joke) < max_length and joke not in seen:
                cleaned.append({"joke": joke})
                seen.add(joke)

    return cleaned

def save_clean_jokes(jokes, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(jokes, f, ensure_ascii=False, indent=2)

# === Run this ===
if __name__ == "__main__":
    raw_dir = "./jokes_raw"
    output_file = "clean_polish_jokes.json"

    raw_jokes = load_all_jokes_from_dir(raw_dir)
    print(f"Loaded {len(raw_jokes)} jokes from raw files.")

    # although I am not sure of that range 5...2500
    clean_jokes = preprocess_jokes(
        raw_jokes,
        min_length=5,
        max_length=2500,
    )
    print(f"After cleaning: {len(clean_jokes)} jokes.")

    save_clean_jokes(clean_jokes, output_file)
    print(f"✅ Clean jokes saved to {output_file}")
