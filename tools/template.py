from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "The key of life is"},
]

prompt_str = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(prompt_str)
