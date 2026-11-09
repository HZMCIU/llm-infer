# LLM Inference (CUDA)

一个从零开始、使用 **CUDA + cuBLAS** 实现的 LLM 推理引擎，目标模型为 **Meta Llama-3.2-1B-Instruct**。

- 纯 C++17 + CUDA，不依赖 PyTorch/libtorch
- 使用 `__nv_bfloat16`（BF16）精度，GEMM 走 `cublasGemmEx`（FP32 累加）
- 实现了完整的 Transformer Decoder 推理流程：Embedding → RMSNorm → GQA Attention（含 RoPE）→ SwiGLU MLP → LM Head → 采样
- 支持 **Prefill + Decode** 两阶段，Decode 阶段使用自定义的 **Paged Attention** CUDA kernel

---

## 目录结构

```
llm-infer/
├── CMakeLists.txt          # 顶层构建脚本
├── include/
│   ├── const.h             # 模型超参 / 常量（硬编码为 Llama-3.2-1B）
│   ├── nn.h                # 权重、层上下文、KV Cache 等结构定义
│   ├── kernel.cuh          # CUDA kernel 函数声明
│   ├── tokenizer.h         # BPE Tokenizer 接口
│   ├── sampler.h           # Top-K / Top-P 采样器接口
│   ├── helper.h            # CUDA_CHECK 宏、GPU 状态检查
│   └── json.hpp            # vendored nlohmann/json（解析 safetensors 头）
├── src/
│   ├── main.cpp            # 入口：加载模型、Prefill + Decode 循环
│   ├── nn.cpp              # 权重加载、Prefill/Decode 计算、采样
│   ├── kernel.cu           # CUDA kernels：RmsNorm/RoPE/Softmax/PagedAttention 等
│   ├── tokenizer.cpp       # BPE 编码/解码
│   ├── sampler.cpp         # Top-K / Top-P 采样
│   └── helper.cpp          # GPU 状态检查
├── test/
│   ├── *.cpp               # GoogleTest 单元测试（kernel/nn/tokenizer/sampler）
│   └── *.py                # PyTorch 对照测试、参考实现
└── tools/
    ├── export.py           # 从 HuggingFace 模型导出 tokenizer.bin
    ├── model-desc.py       # 打印 safetensors 中各张量形状/大小
    ├── tokenizer.py        # 调试 tokenizer 的小脚本
    └── template.py         # 打印 chat template
```

---

## 环境依赖

- **NVIDIA GPU**：计算能力 `sm_86`（Ampere，RTX 30 系；可在 `CMakeLists.txt` 中修改 `CMAKE_CUDA_ARCHITECTURES`）
- **CUDA Toolkit**（含 cuBLAS）
- **CMake >= 3.20**
- **支持 C++17 的编译器**
- （可选）Python + `transformers` / `safetensors`，用于导出 tokenizer 与做对照测试
- GoogleTest 由 CMake 通过 `FetchContent` 自动下载，无需手动安装

---

## 模型准备

代码中的模型路径被硬编码为：

```
/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct
```

该目录需要包含：

```
Llama-3.2-1B-Instruct/
├── model.safetensors   # 权重（BF16）
└── tokenizer.bin       # 由 tools/export.py 生成的自定义 BPE 词表
```

其中 `tokenizer.bin` 由 HuggingFace 原版 tokenizer 导出：

```bash
python tools/export.py
```

> 如果模型放在其它路径，请同步修改 `src/nn.cpp`、`src/main.cpp` 以及 `test/` 下各文件中的路径。

---

## 构建

```bash
# 1. 配置（可指定 GPU 架构，例如 80/86/89）
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86

# 2. 编译
cmake --build build -j

# 3. 运行
./build/main
```

运行后从标准输入读取一行提示词，模型开始自回归生成（最多 300 个 token，遇到 EOS 提前停止）。

---

## 运行测试

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

测试覆盖了 embedding 查表、RMSNorm、RoPE、CausalMask、Softmax、PagedAttention 等 kernel，以及 tokenizer 与 sampler。

---

## 实现说明

### 推理流程

1. **加载权重**（`LoadWeight`）：一次性把整个 `model.safetensors` 读入 GPU 显存，按 header 里的 `data_offsets` 直接以 `__nv_bfloat16*` 指针引用各张量。
2. **Tokenizer**（`tokenizer.cpp`）：实现了与 Llama 3 兼容的 BPE 编码，支持 `<|...|>` 特殊 token。
3. **Prefill**（`TransformerLayersPrefillCompute`）：一次性并行处理整个 prompt，注意力与 MLP 全部用 cuBLAS GEMM 实现。
4. **Decode**（`TransformerLayersDecodeCompute`）：每次只处理 1 个 token，注意力由自定义 `PagedAttention` kernel 完成，配合 **Paged KV Cache**（`block_table` 管理 16 token 为一页的物理块）。
5. **采样**（`sampler.cpp`）：支持 temperature、Top-K（QuickSelect）、Top-P，并保留 argmax 路径。

### 模型超参（`include/const.h`）

| 参数 | 值 |
|------|-----|
| 层数 `N_LAYERS` | 16 |
| 隐藏维度 `EMBEDDING_LENGTH` | 2048 |
| 中间层 `HIDDEN_DIM` | 8192 |
| KV 维度 `KV_DIM` | 512 |
| Head 维度 `HEAD_DIM` | 64 |
| Q / K / V 头数 | 32 / 8 / 8（GQA，比例 4） |
| 词表大小 `VOCAB_SIZE` | 128256 |

### RoPE 参数

在 `nn.cpp` 中调用 `InitRopeFrequency` 时使用 Llama 3.2 的参数：

```
theta=500000, factor=32, low_freq_factor=1, high_freq_factor=4, original_max_position=8192
```

---

## 已知限制 / TODO

- 模型路径与超参全部硬编码，仅支持 Llama-3.2-1B-Instruct，暂未参数化。
- 精度为 BF16，GEMM 使用 FP32 累加；注意力 kernel 内部部分计算在 BF16 下进行，可能与 PyTorch 参考存在轻微数值差异。
- `BATCH_SIZE` 常量已定义，但多请求批处理尚未真正实现。
- 部分缓冲区（如 `attn_scores`）的显存分配按简化方式处理，超长 prompt 下可能越界。
- 采样、KV Cache 页管理等功能仍有较大的健壮性与性能优化空间。
