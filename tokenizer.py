from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/home/yuqyda/Data/cuda/models/LLM-Research--Llama-3.2-1B/snapshots/master"
)
ids = tok.encode("Hello World")
print(ids)
