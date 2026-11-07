from transformers import AutoTokenizer

MODEL_DIR = "/home/yuqyda/Data/cuda/models/LLM-Research--Llama-3.2-1B/snapshots/master"

# local_files_only=True 表示只读本地，不联网
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=False,  # 有些模型（ChatGLM、Qwen 旧版）需要设 True
)

sentence = "Name a color."

# # 只分词，不加 [CLS]/[SEP]/<s> 等特殊符号
# tokens = tokenizer.tokenize(sentence)
# print(tokens)
# # ['Trans', 'former', '的', '分', '词', '器', '把', '句', '子', ...]
#
# # 转成 id
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
#
# # id 转回文本
# print(tokenizer.convert_ids_to_tokens(ids))
# print(tokenizer.decode(ids, skip_special_tokens=True))

# enc = tokenizer(
#     sentence,
#     add_special_tokens=True,  # 是否加 <s> / [CLS] 等
#     return_attention_mask=True,
#     return_token_type_ids=False,  # LLaMA 等不需要
# )
#
# print(enc["input_ids"])  # [[1, 1234, 5678, ...]]
# print(enc["attention_mask"])  # [[1, 1, 1, ...]]
# ids = [
#     128000,
#     128006,
#     882,
#     128007,
#     271,
#     46864,
#     24748,
#     13,
#     128009,
#     128006,
#     78191,
#     128007,
#     271,
# ]
ids = [17, 10544, 10, 128009, 9906, 17, 0, 28, 128009, 60704, 19, 128009, 128009]
text = tokenizer.decode(ids, skip_special_tokens=True)
print(text)
