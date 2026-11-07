import json
import math
import os
import struct


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b"".join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode("utf-8") for ch in token_str
    )


def build_tokenizer(model, output_dir):
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    tokenizer = model.tokenizer

    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())

    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    merge_rank = {
        "".join(tuple(merge if isinstance(merge, list) else merge.split())): i
        for i, merge in enumerate(merges)
    }

    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = 1e-6
        pseudo_scores[token] = score

    max_token_length = max(len(t) for t in all_tokens)
    tokenizer_path = os.path.join(output_dir, "tokenizer.bin")

    with open(tokenizer_path, "wb") as out_f:
        out_f.write(struct.pack("<I", max_token_length))
        out_f.write(struct.pack("<I", model.bos_token_id))
        out_f.write(struct.pack("<I", model.eos_token_id))

        for id, token in enumerate(all_tokens):
            token_bytes = internal_to_bytes(U2B, token)
            out_f.write(struct.pack("f", pseudo_scores[token]))
            out_f.write(struct.pack("<I", len(token_bytes)))
            out_f.write(token_bytes)


def load_tokenizer_and_config(model_path):
    try:
        from transformers import AutoConfig, AutoTokenizer
        from types import SimpleNamespace
    except ImportError as e:
        print(e)
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)

    model_mock = SimpleNamespace()
    model_mock.tokenizer = tokenizer
    model_mock.bos_token_id = (
        hf_config.bos_token_id if hasattr(hf_config, "bos_token_id") else 0
    )
    model_mock.eos_token_id = (
        hf_config.eos_token_id if hasattr(hf_config, "eos_token_id") else 0
    )

    return model_mock


if __name__ == "__main__":
    model_path = (
        "/home/yuqyda/Data/cuda/models/LLM-Research--Llama-3.2-1B/snapshots/master"
    )
    model_info = load_tokenizer_and_config(model_path)

    if model_info:
        build_tokenizer(model_info, model_path)
