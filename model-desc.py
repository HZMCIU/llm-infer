from safetensors import safe_open

path = "/home/yuqyda/Data/cuda/models/LLM-Research--Llama-3.2-1B/snapshots/master/model.safetensors"

# dtype -> 每个元素占用的字节数
DTYPE_SIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

rows = []
with safe_open(path, framework="pt", device="cpu") as f:
    print("metadata:", f.metadata())  # 可能包含 format 等信息
    for name in f.keys():  # 只读 header，不加载张量
        sl = f.get_slice(name)
        shape = sl.get_shape()
        dtype = sl.get_dtype()

        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * DTYPE_SIZE.get(dtype, 0)
        rows.append((name, shape, dtype, numel, nbytes))

# 按参数名排序输出
total_numel = total_bytes = 0
for name, shape, dtype, numel, nbytes in sorted(rows, key=lambda x: x[0]):
    total_numel += numel
    total_bytes += nbytes
    print(
        f"{name:60s} {str(shape):20s} {dtype:8s} "
        f"{numel:>12,} params {nbytes/1024/1024:>10.2f} MB"
    )

print("-" * 100)
print(f"总参数量: {total_numel:,}  ({total_numel/1e9:.2f} B)")
print(f"总大小  : {total_bytes/1024**3:.2f} GB")
