#!/usr/bin/env python3
"""Generate the canonical-GPT-2 PyTorch parity fixture for the cppgpt gate.

This is a DEV-ONLY script: run it in the torch venv to (re)generate
tests/fixtures/gpt2_parity.bin, which is committed so the C++ gate runs under
plain `bazel test` with no torch dependency.

    .venv/bin/python scripts/gen_fixtures.py

It builds a baby canonical GPT-2 (tanh GELU, bias=True, weight-tied head, fp32,
manual attention so the algorithm matches cppgpt op-for-op), then dumps:
  - the model config,
  - the input tokens/targets,
  - the initial parameters, in cppgpt's .bin arena order (grouped by tensor type,
    per-layer tensors stacked as [L, ...]),
  - the logits and per-parameter gradients at the initial weights,
  - the loss after each of 10 AdamW steps (betas 0.9/0.95, eps 1e-8, wd 0).
cppgpt loads the weights/tokens, reruns forward/backward/10 steps, and asserts it
matches to fp32 tolerance (forward 1e-4, gradients 1e-3, loss 1e-3).
"""
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = 0x43475446  # "CGTF"
VERSION = 1

# Baby config — small but exercises multi-layer, multi-head, all ops.
MAX_SEQ_LEN = 16
VOCAB = 17
N_LAYER = 2
N_HEAD = 2
N_EMBD = 8
B, T = 2, 6
N_STEPS = 10
LR = 1e-2
SEED = 1234


class Block(nn.Module):
    def __init__(self, c, nh):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c)
        self.c_attn = nn.Linear(c, 3 * c)
        self.c_proj = nn.Linear(c, c)
        self.ln_2 = nn.LayerNorm(c)
        self.c_fc = nn.Linear(c, 4 * c)
        self.c_projm = nn.Linear(4 * c, c)
        self.nh = nh

    def attn(self, x):
        Bb, Tt, C = x.shape
        hs = C // self.nh
        q, k, v = self.c_attn(x).split(C, dim=2)
        # [B, nh, T, hs]
        q = q.view(Bb, Tt, self.nh, hs).transpose(1, 2)
        k = k.view(Bb, Tt, self.nh, hs).transpose(1, 2)
        v = v.view(Bb, Tt, self.nh, hs).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / (hs ** 0.5))
        mask = torch.tril(torch.ones(Tt, Tt, device=x.device)).view(1, 1, Tt, Tt)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(Bb, Tt, C)
        return self.c_proj(y)

    def mlp(self, x):
        return self.c_projm(F.gelu(self.c_fc(x), approximate="tanh"))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(MAX_SEQ_LEN, N_EMBD)
        self.h = nn.ModuleList([Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

    def forward(self, idx, targets):
        Tt = idx.size(1)
        pos = torch.arange(Tt, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.h:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss


def tensors_in_cppgpt_order(m, grad=False):
    """Params (or their .grad) in cppgpt's .bin arena order."""
    def g(t):
        return t.grad if grad else t

    def stack(sel):
        return torch.stack([g(sel(b)) for b in m.h], dim=0)

    return [
        g(m.wte.weight),                       # wte      [V, C]
        g(m.wpe.weight),                       # wpe      [maxT, C]
        stack(lambda b: b.ln_1.weight),        # ln1w     [L, C]
        stack(lambda b: b.ln_1.bias),          # ln1b
        stack(lambda b: b.c_attn.weight),      # qkvw     [L, 3C, C]
        stack(lambda b: b.c_attn.bias),        # qkvb
        stack(lambda b: b.c_proj.weight),      # attprojw [L, C, C]
        stack(lambda b: b.c_proj.bias),        # attprojb
        stack(lambda b: b.ln_2.weight),        # ln2w
        stack(lambda b: b.ln_2.bias),          # ln2b
        stack(lambda b: b.c_fc.weight),        # fcw      [L, 4C, C]
        stack(lambda b: b.c_fc.bias),          # fcb
        stack(lambda b: b.c_projm.weight),     # fcprojw  [L, C, 4C]
        stack(lambda b: b.c_projm.bias),       # fcprojb
        g(m.ln_f.weight),                      # lnfw
        g(m.ln_f.bias),                        # lnfb
    ]


def flat(tensors):
    return torch.cat([t.reshape(-1).float() for t in tensors]).detach().cpu().numpy()


def main():
    torch.manual_seed(SEED)
    model = GPT()
    idx = torch.randint(0, VOCAB, (B, T))
    targets = torch.randint(0, VOCAB, (B, T))

    params0 = flat(tensors_in_cppgpt_order(model))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.0)
    losses, logits0, grads0 = [], None, None
    for t in range(N_STEPS):
        opt.zero_grad(set_to_none=False)
        logits, loss = model(idx, targets)
        losses.append(loss.item())
        loss.backward()
        if t == 0:
            logits0 = logits.detach().reshape(-1).float().cpu().numpy()
            grads0 = flat(tensors_in_cppgpt_order(model, grad=True))
        opt.step()

    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gpt2_parity.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<9i", MAX_SEQ_LEN, VOCAB, N_LAYER, N_HEAD, N_EMBD, B, T, N_STEPS,
                            int(params0.size)))
        f.write(idx.reshape(-1).int().cpu().numpy().astype("<i4").tobytes())
        f.write(targets.reshape(-1).int().cpu().numpy().astype("<i4").tobytes())
        f.write(params0.astype("<f4").tobytes())
        f.write(logits0.astype("<f4").tobytes())
        f.write(grads0.astype("<f4").tobytes())
        f.write(torch.tensor(losses).numpy().astype("<f4").tobytes())
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    print(f"  config: maxT={MAX_SEQ_LEN} V={VOCAB} L={N_LAYER} H={N_HEAD} C={N_EMBD} B={B} T={T}")
    print(f"  params={params0.size}  logits={logits0.size}  steps={N_STEPS}")
    print(f"  loss trajectory: {[round(x, 4) for x in losses]}")


if __name__ == "__main__":
    main()
