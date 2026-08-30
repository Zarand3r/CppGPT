#!/usr/bin/env python3
"""Generate a tiny synthetic HuggingFace GPT-2 checkpoint for //tools:convert_hf.

WHY THIS EXISTS. convert_hf is the path real GPT-2 weights come through, and it
had no test at all: its two load-bearing behaviours -- the Conv1D transpose and
the 148/160 tensor reconciliation -- were covered only by downloading 522 MB and
eyeballing the result. Everything M-14 claims flows through untested code.

WHY THE VALUES ARE WHAT THEY ARE. Every 2-D weight is filled with

    value[row][col] = row * 1000 + col

so a transposed and an untransposed read are trivially distinguishable: after
conversion, element [out][in] must be `in * 1000 + out`. Random values would
produce a test that passes whether or not the transpose happened, which is the
only bug this file exists to catch. Non-square shapes throughout (3C, 4C), so a
transpose that swaps two equal dimensions cannot hide either.

The causal masks are included deliberately: they are the 12 extra tensors in the
real file (2 here, one per layer) that the reconciliation must skip, and a
fixture without them would not exercise that arithmetic.

Run:  python3 scripts/gen_convert_fixture.py
Writes tests/fixtures/hf_tiny/{model.safetensors,config.json}. Both are small
enough to commit, which is the point -- `bazel test //...` must be green on a
clean checkout without a download.
"""
import json
import pathlib
import struct

N_LAYER, N_HEAD, N_EMBD, N_VOCAB, N_POS = 2, 2, 8, 17, 16
OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "hf_tiny"


def ramp(rows: int, cols: int) -> bytes:
    """row*1000 + col, row-major -- so a transpose is visible in the values."""
    return b"".join(
        struct.pack("<f", float(r * 1000 + c)) for r in range(rows) for c in range(cols)
    )


def flat(n: int, base: float) -> bytes:
    return b"".join(struct.pack("<f", base + i) for i in range(n))


def main() -> int:
    C, V, P = N_EMBD, N_VOCAB, N_POS
    tensors: dict[str, tuple[list[int], bytes]] = {
        "wte.weight": ([V, C], ramp(V, C)),
        "wpe.weight": ([P, C], ramp(P, C)),
        "ln_f.weight": ([C], flat(C, 1.0)),
        "ln_f.bias": ([C], flat(C, 0.0)),
    }
    for l in range(N_LAYER):
        h = f"h.{l}."
        tensors[h + "ln_1.weight"] = ([C], flat(C, 1.0))
        tensors[h + "ln_1.bias"] = ([C], flat(C, 0.0))
        tensors[h + "ln_2.weight"] = ([C], flat(C, 1.0))
        tensors[h + "ln_2.bias"] = ([C], flat(C, 0.0))
        # HF Conv1D stores [in, out]. These are the four transposed tensors.
        tensors[h + "attn.c_attn.weight"] = ([C, 3 * C], ramp(C, 3 * C))
        tensors[h + "attn.c_attn.bias"] = ([3 * C], flat(3 * C, 0.0))
        tensors[h + "attn.c_proj.weight"] = ([C, C], ramp(C, C))
        tensors[h + "attn.c_proj.bias"] = ([C], flat(C, 0.0))
        tensors[h + "mlp.c_fc.weight"] = ([C, 4 * C], ramp(C, 4 * C))
        tensors[h + "mlp.c_fc.bias"] = ([4 * C], flat(4 * C, 0.0))
        tensors[h + "mlp.c_proj.weight"] = ([4 * C, C], ramp(4 * C, C))
        tensors[h + "mlp.c_proj.bias"] = ([C], flat(C, 0.0))
        # The causal mask: a tensor the file carries and convert_hf must SKIP.
        tensors[h + "attn.bias"] = ([1, 1, P, P], b"\x00\x00\x80\x3f" * (P * P))

    header: dict[str, object] = {}
    offset = 0
    for name, (shape, blob) in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)

    hdr = json.dumps(header, separators=(",", ":")).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)  # safetensors pads the header to 8 bytes

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for _, blob in tensors.values():
            f.write(blob)

    (OUT / "config.json").write_text(
        json.dumps(
            {
                "n_layer": N_LAYER,
                "n_head": N_HEAD,
                "n_embd": N_EMBD,
                "vocab_size": N_VOCAB,
                "n_positions": N_POS,
            },
            indent=2,
        )
        + "\n"
    )
    n_param = len(tensors) - N_LAYER
    print(f"  {len(tensors)} tensors ({n_param} parameters + {N_LAYER} causal masks)")
    print(f"  wrote {OUT}/model.safetensors and config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
