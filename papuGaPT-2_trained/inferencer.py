from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("flax-community/papuGaPT2", load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, "polish-joke-gpt-qlora")

tokenizer = AutoTokenizer.from_pretrained("flax-community/papuGaPT2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2-style

input_text = "Wilk pyta sie"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(output[0], skip_special_tokens=True))
