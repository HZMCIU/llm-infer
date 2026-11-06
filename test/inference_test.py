import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import cosine
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import torch.nn.functional as F

local_model_path = (
    "/home/yuqyda/Data/cuda/models/LLM-Research--Llama-3.2-1B/snapshots/master"
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    attn_implementation="eager",
)

model.eval()

prompt = "What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


def compare_prefill_output():
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]
        kv_cache = outputs.past_key_values

        next_token_id = logits[:, -1, :].argmax(dim=-1)
        # print("Prefill 预测的第一个词:", tokenizer.decode(next_token_id))
        np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=6)
        torch.set_printoptions(
            threshold=float("inf"),
            sci_mode=False,
            precision=6,
            linewidth=200,
        )

        a = np.loadtxt("a.txt", dtype=np.float32)
        # b = last_hidden[0, :, :].detach().float().cpu().numpy()
        b = logits[0, -1, :].detach().float().cpu().numpy()
        np.savetxt("b.txt", b, delimiter=" ", fmt="%.6f")
        # print(1 - cosine(b.flatten(), a.flatten()))
        print(logits[0, -1, :].argmax())


# 假设 model 和 inputs 已经准备好 (且加载时设置了 attn_implementation="eager")


def save_and_compare(b):
    a = np.loadtxt("a.txt", dtype=np.float32)
    b = b[0, :, :].detach().float().cpu().numpy()
    np.savetxt("b.txt", b.flatten(), delimiter=" ", fmt="%.6f")
    print(1 - cosine(b.flatten(), a.flatten()))


def one_layer_compute():
    layer0 = model.model.layers[0]
    with torch.no_grad():
        # =================================================================
        # Step 0: Embedding 输出 (Layer 0 的输入) -> [B, S, 4096]
        # =================================================================
        h0_embed = model.model.embed_tokens(inputs["input_ids"])
        # =================================================================
        # Step 1: Input RMSNorm -> [B, S, 4096]
        # =================================================================
        h1_norm1 = layer0.input_layernorm(h0_embed)

        # =================================================================
        # Step 2: Q, K, V 线性投影 (未拆头、未加 RoPE)
        # =================================================================
        q_raw = layer0.self_attn.q_proj(h1_norm1)  # [B, S, 4096]
        k_raw = layer0.self_attn.k_proj(h1_norm1)  # [B, S, 1024 (GQA) 或 4096 (MHA)]
        v_raw = layer0.self_attn.v_proj(h1_norm1)  # [B, S, 1024 (GQA) 或 4096 (MHA)]

        # =================================================================
        # Step 3: 拆头并转置为 [B, Num_Heads, S, Head_Dim]
        # =================================================================
        bsz, q_len, _ = h1_norm1.size()
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // num_heads

        q = q_raw.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = k_raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v_raw.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # =================================================================
        # Step 4: RoPE (旋转位置编码) —— 【最高频出错点！】
        # =================================================================
        position_ids = torch.arange(q_len, device=model.device).unsqueeze(0)
        # 获取 cos, sin (兼容新版和旧版 transformers)
        if hasattr(model.model, "rotary_emb"):
            cos, sin = model.model.rotary_emb(v, position_ids)
        else:
            cos, sin = layer0.self_attn.rotary_emb(v, position_ids)

        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        # save_and_compare(k_rope.transpose(1, 2).contiguous())

        # ⚠️ 注意：如果你在 C++ 里此时还没把 head 维度转置到前面，即形状还是 [B, S, H, D]，
        # 请对比 q_rope.transpose(1, 2).contiguous() ！

        # =================================================================
        # Step 5: Attention 计算 (GQA广播 -> Q@K^T -> Mask -> Softmax -> @V)
        # =================================================================
        num_key_value_groups = num_heads // num_kv_heads
        k_rep = repeat_kv(k_rope, num_key_value_groups)  # [B, 32, S, 128]
        v_rep = repeat_kv(v, num_key_value_groups)  # [B, 32, S, 128]

        # 5.1 Attention Scores (Q @ K^T / sqrt(d)) -> [B, 32, S, S]
        attn_weights = torch.matmul(q_rope, k_rep.transpose(2, 3)) / (head_dim**0.5)
        # save_and_compare(attn_weights.flatten(1, 2))

        # 5.2 加上下三角 Causal Mask (上三角设为极小负数)
        causal_mask = torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=model.device), diagonal=1
        )
        attn_weights_masked = attn_weights + causal_mask
        # save_and_compare(attn_weights_masked)

        # 5.3 Softmax (注意 HF 内部强制在 fp32 下算完再转回 bf16/fp16)
        attn_probs = F.softmax(attn_weights_masked, dim=-1, dtype=torch.float32).to(
            q.dtype
        )

        # 5.4 乘 V 并合并多头 -> [B, S, 4096]
        attn_out_heads = torch.matmul(attn_probs, v_rep)  # [B, 32, S, 128]
        attn_out_merged = (
            attn_out_heads.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        )

        # =================================================================
        # Step 6: O_Proj 输出投影与第一次残差连接 -> [B, S, 4096]
        # =================================================================
        o_proj_out = layer0.self_attn.o_proj(attn_out_merged)
        # save_and_compare(o_proj_out)
        h2_after_attn = h0_embed + o_proj_out  # 第一次残差相加！
        # save_and_compare(h2_after_attn)
        # save_and_compare(h2_after_attn)

        # =================================================================
        # Step 7: Post-Attention RMSNorm -> [B, S, 4096]
        # =================================================================
        h3_norm2 = layer0.post_attention_layernorm(h2_after_attn)
        # save_and_compare(h3_norm2)

        # =================================================================
        # Step 8: MLP (SwiGLU) 与第二次残差连接 -> [B, S, 4096]
        # =================================================================
        gate = layer0.mlp.gate_proj(h3_norm2)  # [B, S, Inter_Size]
        up = layer0.mlp.up_proj(h3_norm2)  # [B, S, Inter_Size]
        act = F.silu(gate) * up  # SwiGLU 激活相乘
        mlp_out = layer0.mlp.down_proj(act)  # [B, S, 4096]

        layer0_final_out = (
            h2_after_attn + mlp_out
        )  # 第二次残差相加 (即 outputs.hidden_states[1])
        save_and_compare(layer0_final_out)


def regressive_decode():
    input_len = inputs["input_ids"].shape[1]  # 记录输入 Prompt 的 Token 长度
    with torch.no_grad():
        # max_new_tokens=10: 严格生成 10 个新 Token
        # do_sample=False: 关闭随机采样，使用确定性的 argmax (Greedy Search)
        generated_sequences = model.generate(
            **inputs,
            max_new_tokens=10,
            min_new_tokens=10,  # 防止遇到 EOS 提前终止，强制生成满 10 个
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_ids = generated_sequences[0].tolist()

    new_10_token_ids = generated_sequences[0, input_len:].tolist()

    print("新生成的 10 个 Token ID:", new_10_token_ids)
    print(
        "每个 ID 对应的单词     :",
        [tokenizer.decode([tid]) for tid in new_10_token_ids],
    )
    print("拼接后的完整生成文本   :", repr(tokenizer.decode(new_10_token_ids)))


if __name__ == "__main__":
    regressive_decode()
