"""Create a tiny deterministic llama-architecture F32 GGUF for ROCmFP4 golden data.

Emits small 2D tensors only; dimensions are multiples of 32 so ROCmFP4 blocks
tile exactly. The RNG is a fixed PCG64 seed so the byte stream is reproducible.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"H:\ports\ROCmFPX\gguf-py")
from gguf import GGUFWriter  # noqa: E402

OUT = Path(sys.argv[1])
SEED = 20260923

rng = np.random.Generator(np.random.PCG64(SEED))

N_EMBD = 256
N_FF = 512
N_HEADS = 8
N_LAYERS = 2
VOCAB = 1024

gw = GGUFWriter(None, "llama")

gw.add_name("rocmfp4-tiny-src")
gw.add_context_length(2048)
gw.add_embedding_length(N_EMBD)
gw.add_block_count(N_LAYERS)
gw.add_feed_forward_length(N_FF)
gw.add_head_count(N_HEADS)
gw.add_head_count_kv(N_HEADS)
gw.add_layer_norm_rms_eps(1e-5)
gw.add_rope_dimension_count(N_EMBD // N_HEADS)
gw.add_file_type(0)


def tensor(name, shape):
    # Centered, roughly unit-variance weights with a few deliberate outliers so
    # the upstream scale search exercises both tail and clip paths.
    a = rng.standard_normal(shape, dtype=np.float32)
    a[0, 0] = 12.5
    a[-1, -1] = -9.75
    if a.shape[0] > 1 and a.shape[1] > 1:
        a[1, 1] = 0.0
    return a.astype(np.float32)


gw.add_tokenizer_model("llama")
gw.add_tokenizer_pre("default")
gw.add_token_list([f"tok{i}" for i in range(VOCAB)])
gw.add_token_scores([0.0] * VOCAB)
gw.add_token_types([1] * VOCAB)

gw.add_tensor("token_embd.weight", tensor("token_embd.weight", (N_EMBD, VOCAB)))
for il in range(N_LAYERS):
    p = f"blk.{il}"
    gw.add_tensor(f"{p}.attn_norm.weight", np.abs(tensor(f"{p}.attn_norm.weight", (N_EMBD, 1))) + 0.5)
    gw.add_tensor(f"{p}.attn_q.weight", tensor(f"{p}.attn_q.weight", (N_EMBD, N_EMBD)))
    gw.add_tensor(f"{p}.attn_k.weight", tensor(f"{p}.attn_k.weight", (N_EMBD, N_EMBD // 2)))
    gw.add_tensor(f"{p}.attn_v.weight", tensor(f"{p}.attn_v.weight", (N_EMBD, N_EMBD // 2)))
    gw.add_tensor(f"{p}.attn_output.weight", tensor(f"{p}.attn_output.weight", (N_EMBD, N_EMBD)))
    gw.add_tensor(f"{p}.ffn_norm.weight", np.abs(tensor(f"{p}.ffn_norm.weight", (N_EMBD, 1))) + 0.5)
    gw.add_tensor(f"{p}.ffn_gate.weight", tensor(f"{p}.ffn_gate.weight", (N_EMBD, N_FF)))
    gw.add_tensor(f"{p}.ffn_up.weight", tensor(f"{p}.ffn_up.weight", (N_EMBD, N_FF)))
    gw.add_tensor(f"{p}.ffn_down.weight", tensor(f"{p}.ffn_down.weight", (N_FF, N_EMBD)))
gw.add_tensor("output_norm.weight", np.abs(tensor("output_norm.weight", (N_EMBD, 1))) + 0.5)
gw.add_tensor("output.weight", tensor("output.weight", (N_EMBD, VOCAB)))

gw.write_header_to_file(OUT)
gw.write_kv_data_to_file()
gw.write_ti_data_to_file()
while gw.tensors[0]:
    gw.write_tensor_data(next(iter(gw.tensors[0].values())).tensor)
gw.close()

print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
