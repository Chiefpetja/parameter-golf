"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))


        self.rotary = Rotary(self.head_dim, base=rope_base)

        # High-strength visibility path
        self.strength_q = nn.Linear(1, num_heads, bias=True)
        self.strength_k = nn.Linear(1, num_kv_heads, bias=True)

        with torch.no_grad():
            self.strength_q.weight.zero_()
            self.strength_q.bias.zero_()
            self.strength_k.weight.zero_()
            self.strength_k.bias.zero_()

        # --- LOW-TAU SOFT DESCRIPTOR OVERLAY ---
        # Gate sees a richer local context and gets a tiny learnable artifact space
        # before producing the final selective gate.
        self.gate_feature_proj = nn.Linear(3, 8, bias=True)
        self.role_gate_proj = nn.Linear(8, self.num_kv_heads, bias=True)
        
        # Small descriptor space: not a fixed role, just a soft extra coordinate system
        self.low_descriptor_proj = nn.Linear(1, 4 * self.num_kv_heads, bias=True)
        self.descriptor_to_k = nn.Linear(4 * self.num_kv_heads, self.num_kv_heads * self.head_dim, bias=False)
        self.descriptor_to_v = nn.Linear(4 * self.num_kv_heads, self.num_kv_heads * self.head_dim, bias=False)

        self.v_role_scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.k_role_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # --- STRUCTURE CLUSTERING (K=12) ---
        self.cluster_k = 12
        self.cluster_proj = nn.Linear(4 * self.num_kv_heads, self.cluster_k, bias=True)

        with torch.no_grad():
            nn.init.normal_(self.cluster_proj.weight, mean=0.0, std=0.02)
            self.cluster_proj.bias.zero_()

        # Additional learned low-tau scaling so the low path can grow into
        # real competition with the high-tau path, separately for K and V.
        self.low_k_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.low_v_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Minimal descriptor/cluster diagnostics
        self.obs_low_descriptor_sum: float = 0.0
        self.obs_role_gate_sum: float = 0.0
        self.obs_role_v_overlay_sum: float = 0.0
        self.obs_cluster_usage_max_sum: float = 0.0
        self.obs_cluster_usage_entropy_sum: float = 0.0

        # Extra cluster-drift diagnostics
        self.obs_cluster_presence_max_sum: float = 0.0
        self.obs_cluster_presence_entropy_sum: float = 0.0
        self.obs_token_overlay_mean_sum: float = 0.0
        self.obs_region_bridge_abs_sum: float = 0.0

        self.obs_role_diag_count: int = 0

        with torch.no_grad():
            nn.init.normal_(self.gate_feature_proj.weight, mean=0.0, std=0.02)
            self.gate_feature_proj.bias.zero_()

            self.role_gate_proj.weight.zero_()
            self.role_gate_proj.bias.fill_(-2.0)  # start almost off

            nn.init.normal_(self.low_descriptor_proj.weight, mean=0.0, std=0.02)
            self.low_descriptor_proj.bias.zero_()

            nn.init.normal_(self.descriptor_to_k.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.descriptor_to_v.weight, mean=0.0, std=0.02)

    def reset_role_diagnostics(self) -> None:
        self.obs_low_descriptor_sum = 0.0
        self.obs_role_gate_sum = 0.0
        self.obs_role_v_overlay_sum = 0.0
        self.obs_cluster_usage_max_sum = 0.0
        self.obs_cluster_usage_entropy_sum = 0.0

        self.obs_cluster_presence_max_sum = 0.0
        self.obs_cluster_presence_entropy_sum = 0.0
        self.obs_token_overlay_mean_sum = 0.0
        self.obs_region_bridge_abs_sum = 0.0

        self.obs_role_diag_count = 0

    def get_role_diagnostics(self) -> dict[str, float]:
        if self.obs_role_diag_count == 0:
            return {
                "low_descriptor_mean": 0.0,
                "role_gate_mean": 0.0,
                "role_v_overlay_mean": 0.0,
                "cluster_usage_max": 0.0,
                "cluster_usage_entropy": 0.0,
                "cluster_presence_max": 0.0,
                "cluster_presence_entropy": 0.0,
                "token_overlay_mean": 0.0,
                "region_bridge_abs_mean": 0.0,
                "count": 0.0,
            }

        return {
            "low_descriptor_mean": self.obs_low_descriptor_sum / self.obs_role_diag_count,
            "role_gate_mean": self.obs_role_gate_sum / self.obs_role_diag_count,
            "role_v_overlay_mean": self.obs_role_v_overlay_sum / self.obs_role_diag_count,
            "cluster_usage_max": self.obs_cluster_usage_max_sum / self.obs_role_diag_count,
            "cluster_usage_entropy": self.obs_cluster_usage_entropy_sum / self.obs_role_diag_count,
            "cluster_presence_max": self.obs_cluster_presence_max_sum / self.obs_role_diag_count,
            "cluster_presence_entropy": self.obs_cluster_presence_entropy_sum / self.obs_role_diag_count,
            "token_overlay_mean": self.obs_token_overlay_mean_sum / self.obs_role_diag_count,
            "region_bridge_abs_mean": self.obs_region_bridge_abs_sum / self.obs_role_diag_count,
            "count": float(self.obs_role_diag_count),
        }
    
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        # Default token-local strength proxy
        token_energy = x.abs().mean(dim=-1, keepdim=True)  # [B, T, 1]
        strength_tok = token_energy
        if hasattr(self, "last_strength_tok"):
            strength_tok = self.last_strength_tok  # [B, T, 1]

        # --- LOW-TAU SIGNAL ---
        low_score = torch.exp(-1.0 * strength_tok)  # [B, T, 1]

        # Gate sees a richer local context:
        # low_score      -> "how low-tau / quiet"
        # strength_tok   -> direct tension signal
        # token_energy   -> current local activity proxy from x
        gate_input = torch.cat(
            [low_score, strength_tok, token_energy],
            dim=-1,
        )  # [B, T, 3]

        gate_hidden = torch.tanh(
            self.gate_feature_proj(gate_input)
        )  # [B, T, 8]

        role_gate = torch.sigmoid(
            self.role_gate_proj(gate_hidden)
        )  # [B, T, KVH]

        # Soft descriptor instead of fixed role assignment
        low_descriptor = torch.tanh(
            self.low_descriptor_proj(low_score)
        )  # [B, T, 4*KVH]

        # --- SOFT CLUSTERING OVER DESCRIPTION ---
        cluster_logits = self.cluster_proj(low_descriptor)  # [B, T, K]
        cluster_weights = torch.softmax(cluster_logits, dim=-1)  # [B, T, K]

        # expose for refine
        self.last_cluster_weights = cluster_weights  # [B, T, K]

        # -----------------------------
        # Descriptor + cluster confidence export for refine
        # -----------------------------
        eps = 1e-8

        # Token-local cluster certainty
        cluster_confidence_tok = cluster_weights.max(dim=-1, keepdim=True).values  # [B,T,1]

        # Token-local cluster entropy
        cluster_entropy_tok = -(
            cluster_weights * (cluster_weights + eps).log()
        ).sum(dim=-1, keepdim=True)  # [B,T,1]

        # Regional descriptor means
        region_descriptor_sum = torch.einsum(
            "btk,btd->bkd",
            cluster_weights,
            low_descriptor,
        )  # [B,K,D_desc]

        cluster_sizes = cluster_weights.sum(dim=1).clamp_min(eps)  # [B,K]

        region_descriptor = region_descriptor_sum / cluster_sizes.unsqueeze(-1)

        # Backproject descriptor to token space
        token_descriptor = torch.einsum(
            "btk,bkd->btd",
            cluster_weights,
            region_descriptor,
        )  # [B,T,D_desc]

        # expose for refine
        self.last_token_descriptor = token_descriptor
        self.last_cluster_confidence_tok = cluster_confidence_tok
        self.last_cluster_entropy_tok = cluster_entropy_tok

        role_k_overlay = self.descriptor_to_k(low_descriptor).reshape(
            bsz, seqlen, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [B, KVH, T, D]

        role_v_overlay = self.descriptor_to_v(low_descriptor).reshape(
            bsz, seqlen, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [B, KVH, T, D]

        with torch.no_grad():
            self.obs_low_descriptor_sum += float(low_descriptor.abs().mean().item())
            self.obs_role_gate_sum += float(role_gate.mean().item())
            self.obs_role_v_overlay_sum += float(role_v_overlay.abs().mean().item())
            self.obs_token_overlay_mean_sum += float(role_v_overlay.abs().mean(dim=(1, 3)).mean().item())
            self.obs_role_diag_count += 1

        # --- PROJECTIONS ---
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # --- NORMALIZATION ---
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # --- HIGH-STRENGTH VISIBILITY ---
        q_strength = 1.0 + 0.5 * torch.tanh(
            self.strength_q(strength_tok).to(dtype=q.dtype)
        )  # [B, T, H]

        k_strength = 1.0 + 0.5 * torch.tanh(
            self.strength_k(strength_tok).to(dtype=k.dtype)
        )  # [B, T, KVH]

        q = q * q_strength.transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]
        k = k * k_strength.transpose(1, 2).unsqueeze(-1)  # [B, KVH, T, 1]

        # --- LOW-TAU SOFT OVERLAY ---
        role_gate = role_gate.transpose(1, 2).unsqueeze(-1)  # [B, KVH, T, 1]

        # Expose tiny token-local structure summaries for downstream refine use
        self.last_role_gate_tok = role_gate.mean(dim=1)  # [B, T, 1]
        self.last_role_context_tok = role_v_overlay.abs().mean(dim=(1, 3), keepdim=False).unsqueeze(-1)  # [B, T, 1]

        # overlay strength per token (clean scalar)
        overlay_tok = role_v_overlay.abs().mean(dim=(1, 3))  # [B, T]
        self.last_overlay_tok = overlay_tok.unsqueeze(-1)    # [B, T, 1]

        # debug: cluster usage
        with torch.no_grad():
            self.last_cluster_usage = cluster_weights.mean(dim=(0, 1))  # [K]

            cluster_usage = self.last_cluster_usage
            cluster_usage_max = cluster_usage.max()

            # small epsilon for safe log
            eps = 1e-8
            cluster_usage_entropy = -(cluster_usage * (cluster_usage + eps).log()).sum()

            self.obs_cluster_usage_max_sum += float(cluster_usage_max.item())
            self.obs_cluster_usage_entropy_sum += float(cluster_usage_entropy.item())

            # Per-batch cluster presence diagnostics.
            # Shows whether clusters collapse globally or only within samples.
            cluster_presence = cluster_weights.mean(dim=1)  # [B, K]
            cluster_presence_max = cluster_presence.max(dim=-1).values.mean()

            eps = 1e-8
            cluster_presence_entropy = -(
                cluster_presence * (cluster_presence + eps).log()
            ).sum(dim=-1).mean()

            self.obs_cluster_presence_max_sum += float(cluster_presence_max.item())
            self.obs_cluster_presence_entropy_sum += float(cluster_presence_entropy.item())

            # Approximate bridge-region magnitude using normalized region means.
            cluster_sizes = cluster_weights.sum(dim=1).clamp_min(eps)  # [B, K]
            region_bridge_sum = torch.einsum(
                "btk,btd->bkd",
                cluster_weights,
                x.detach(),
            )
            region_bridge = region_bridge_sum / cluster_sizes.unsqueeze(-1)
            self.obs_region_bridge_abs_sum += float(region_bridge.abs().mean().item())
            
        k = k + (
            role_gate
            * self.low_k_scale.to(dtype=k.dtype)
            * self.k_role_scale.to(dtype=k.dtype)
            * role_k_overlay.to(dtype=k.dtype)
        )

        v = v + (
            role_gate
            * self.low_v_scale.to(dtype=v.dtype)
            * self.v_role_scale.to(dtype=v.dtype)
            * role_v_overlay.to(dtype=v.dtype)
        )

        # --- ATTENTION ---
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Dual-space relu^2 MLP with low-rank diagonal soft parity modulation.
    def __init__(self, dim: int, mlp_mult: int, descriptor_dim: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.hidden = hidden

        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.neg_beta: float = 1.0
        self.dual_alpha: float = 0.0  # trainer-controlled: 0.0 early, 1.0 after switch

        # Token-local delta control from tension energy
        self.delta_gate_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.calc_delta_alpha = nn.Linear(1, 1, bias=True)
        self.calc_delta_tau = nn.Linear(1, 1, bias=True)
        self.calc_base_keep = nn.Linear(1, 1, bias=True)

        with torch.no_grad():
            self.calc_delta_alpha.weight.fill_(0.0)
            self.calc_delta_alpha.bias.fill_(0.30)

            self.calc_delta_tau.weight.fill_(0.0)
            self.calc_delta_tau.bias.fill_(-0.05)

            self.calc_base_keep.weight.fill_(0.0)
            self.calc_base_keep.bias.fill_(0.85)

        # Low-rank diagonal soft parity settings
        self.parity_rank: int = 16
        self.parity_alpha: float = 0.05
        self.mode_in = CastedLinear(dim, self.parity_rank, bias=False)
        self.mode_out = CastedLinear(self.parity_rank, hidden, bias=False)

        # Wider semantic refine channel.
        # Keeps bridge information less bottlenecked before projecting back to hidden.
        self.bridge_dim: int = 128
        self.refine_dim: int = 64

        # refine input:
        # [strength_tok, attn_context_tok, overlay_tok] + translated cluster-bridge signal [bridge_dim]
        self.refine_in = nn.Linear(self.bridge_dim + 3, self.refine_dim, bias=True)
        self.refine_to_hidden = nn.Linear(self.refine_dim, hidden, bias=False)
        self.refine_use_proj = nn.Linear(self.refine_dim, 1, bias=True)

        # Cluster bridge now receives:
        # [token_bridge,
        #  h_base,
        #  bridge_delta,
        #  strength_tok,
        #  token_descriptor,
        #  cluster_confidence_tok,
        #  cluster_entropy_tok]
        #
        # = [hidden + hidden + hidden + 1 + descriptor_dim + 2]

        self.descriptor_dim = descriptor_dim

        self.bridge_translator = nn.Linear(
            (3 * hidden) + 1 + self.descriptor_dim + 2,
            self.bridge_dim,
            bias=True,
        )
        # explicit base/refine mixing:
        # input = [tau/strength, attractor_hint]
        # output = token-local permission to move away from h_base
        self.base_mix_in = nn.Linear(2, 8, bias=True)
        self.base_mix_out = nn.Linear(8, 1, bias=True)

        # token-local dual refine scaling:
        # input = [tau/strength, attractor_hint]
        # output = two token-local scales:
        # - local refine path
        # - attractor path
        self.dual_refine_scale_in = nn.Linear(2, 8, bias=True)
        self.dual_refine_scale_local = nn.Linear(8, 1, bias=True)
        self.dual_refine_scale_attr = nn.Linear(8, 1, bias=True)

        # --- cluster-based attractor path ---
        self.cluster_attr_proj = nn.Linear(hidden, hidden, bias=False)
        self.attr_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # --- connection-state path ---
        # combines:
        # - h_base
        # - h_base_shifted
        # - h_refine
        # - token-local tension
        self.connection_in = nn.Linear((3 * hidden) + 1, hidden, bias=True)

        # recovery head:
        # brings back useful base-linked predictability information into h_refine
        self.recovery_proj = nn.Linear(hidden, hidden, bias=False)
        self.recovery_gate = nn.Linear(hidden, 1, bias=True)

        # saturation vector path:
        # creates learned return vectors that refold distant refine tension
        # back into a base-compatible / predictable representation.
        self.saturation_gate = nn.Linear(hidden, 1, bias=True)
        self.saturation_vec_proj = nn.Linear(hidden, hidden, bias=False)

        self.recovery_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        with torch.no_grad():
            nn.init.normal_(self.cluster_attr_proj.weight, mean=0.0, std=0.02)

            nn.init.normal_(self.connection_in.weight, mean=0.0, std=0.02)
            self.connection_in.bias.zero_()

            nn.init.normal_(self.recovery_proj.weight, mean=0.0, std=0.02)

            self.recovery_gate.weight.zero_()
            self.recovery_gate.bias.fill_(-2.0)  # conservative start

            nn.init.normal_(self.saturation_gate.weight, mean=0.0, std=0.02)
            self.saturation_gate.bias.fill_(-1.0)

            nn.init.normal_(self.saturation_vec_proj.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            nn.init.normal_(self.refine_in.weight, mean=0.0, std=0.02)
            self.refine_in.bias.zero_()
            nn.init.normal_(self.refine_to_hidden.weight, mean=0.0, std=0.02)

            nn.init.normal_(self.bridge_translator.weight, mean=0.0, std=0.02)
            self.bridge_translator.bias.zero_()

            self.refine_use_proj.weight.zero_()
            self.refine_use_proj.bias.fill_(-2.0)  # starts very conservative

            nn.init.normal_(self.base_mix_in.weight, mean=0.0, std=0.02)
            self.base_mix_in.bias.zero_()

            self.base_mix_out.weight.zero_()
            self.base_mix_out.bias.fill_(-0.5)  # starts partially open: sigmoid(-0.5) ~= 0.38

            nn.init.normal_(self.dual_refine_scale_in.weight, mean=0.0, std=0.02)
            self.dual_refine_scale_in.bias.zero_()

            self.dual_refine_scale_local.weight.zero_()
            self.dual_refine_scale_local.bias.fill_(-1.5)

            self.dual_refine_scale_attr.weight.zero_()
            self.dual_refine_scale_attr.bias.fill_(-1.5)

        # caches from base/refine passes
        self.last_strength_tok: Tensor | None = None
        self.last_h_base: Tensor | None = None
        self.last_predictability_signature_tok: Tensor | None = None
        self.last_saturation_tok: Tensor | None = None

        # minimal refine diagnostics
        self.obs_refine_use_sum: float = 0.0
        self.obs_delta_h_sum: float = 0.0
        self.obs_scale_local_sum: float = 0.0
        self.obs_scale_attr_sum: float = 0.0
        self.obs_recovery_gate_sum: float = 0.0
        self.obs_saturation_gate_sum: float = 0.0
        self.obs_saturation_vec_sum: float = 0.0
        self.obs_refine_tension_sum: float = 0.0
        self.obs_refine_diag_count: int = 0

    def reset_refine_diagnostics(self) -> None:
        self.obs_refine_use_sum = 0.0
        self.obs_delta_h_sum = 0.0
        self.obs_scale_local_sum = 0.0
        self.obs_scale_attr_sum = 0.0
        self.obs_recovery_gate_sum = 0.0
        self.obs_saturation_gate_sum = 0.0
        self.obs_saturation_vec_sum = 0.0
        self.obs_refine_tension_sum = 0.0
        self.obs_refine_diag_count = 0

    def get_refine_diagnostics(self) -> dict[str, float]:
        if self.obs_refine_diag_count == 0:
            return {
                "refine_use_mean": 0.0,
                "delta_h_mean": 0.0,
                "scale_local_mean": 0.0,
                "scale_attr_mean": 0.0,
                "recovery_gate_mean": 0.0,
                "saturation_gate_mean": 0.0,
                "saturation_vec_mean": 0.0,
                "refine_tension_mean": 0.0,
                "count": 0.0,
            }

        return {
            "refine_use_mean": self.obs_refine_use_sum / self.obs_refine_diag_count,
            "delta_h_mean": self.obs_delta_h_sum / self.obs_refine_diag_count,
            "scale_local_mean": self.obs_scale_local_sum / self.obs_refine_diag_count,
            "scale_attr_mean": self.obs_scale_attr_sum / self.obs_refine_diag_count,
            "recovery_gate_mean": self.obs_recovery_gate_sum / self.obs_refine_diag_count,
            "saturation_gate_mean": self.obs_saturation_gate_sum / self.obs_refine_diag_count,
            "saturation_vec_mean": self.obs_saturation_vec_sum / self.obs_refine_diag_count,
            "refine_tension_mean": self.obs_refine_tension_sum / self.obs_refine_diag_count,
            "count": float(self.obs_refine_diag_count),
        }

    def forward_base(self, x: Tensor) -> Tensor:
        u = self.fc(x)

        # Low-rank diagonal soft parity modulation
        mode_latent = self.mode_in(x)
        mode = torch.tanh(self.mode_out(mode_latent))
        u = u * (1.0 + self.parity_alpha * mode)

        pos = torch.relu(u).square()
        neg = torch.relu(-u).square()

        base = pos + (self.dual_alpha * self.neg_beta) * neg
        delta = pos - neg

        # Token-local tension summary -> token-local reaction values
        strength_tok = delta.abs().mean(dim=-1, keepdim=True)
        self.last_strength_tok = strength_tok.detach()

        alpha_tok = self.calc_delta_alpha(strength_tok)
        tau_tok = self.calc_delta_tau(strength_tok)
        base_keep_tok = torch.sigmoid(self.calc_base_keep(strength_tok))

        strength = delta.abs()
        gate = torch.sigmoid(
            self.delta_gate_scale.to(dtype=pos.dtype) * (
                strength - tau_tok.to(dtype=pos.dtype)
            )
        )

        carry = alpha_tok.to(dtype=pos.dtype) * gate * delta
        h_base = base_keep_tok.to(dtype=pos.dtype) * base + carry

        self.last_h_base = h_base
        return self.proj(h_base)

    def forward_refine(
        self,
        attn_weight_tok: Tensor | None,
        attn_context_tok: Tensor | None,
        cluster_weights: Tensor | None,
        overlay_tok: Tensor | None,
        token_descriptor: Tensor | None,
        cluster_confidence_tok: Tensor | None,
        cluster_entropy_tok: Tensor | None,
    ) -> Tensor | None:
        if self.last_h_base is None or self.last_strength_tok is None:
            return None
        if attn_weight_tok is None or attn_context_tok is None:
            return None
        if (
            cluster_weights is None
            or overlay_tok is None
            or token_descriptor is None
            or cluster_confidence_tok is None
            or cluster_entropy_tok is None
        ):
            return None
        
        h_base = self.last_h_base
        strength_tok = self.last_strength_tok.to(dtype=h_base.dtype)

        # refine is now the actual transfer path into parameter space,
        # so it may learn through the prepared representation space
        cluster_weights = cluster_weights.to(dtype=h_base.dtype)   # [B, T, K]
        overlay_tok = overlay_tok.to(dtype=h_base.dtype)           # [B, T, 1]
        attn_weight_tok = attn_weight_tok.to(dtype=h_base.dtype)   # [B, T, 1]
        attn_context_tok = attn_context_tok.to(dtype=h_base.dtype) # [B, T, 1]
        token_descriptor = token_descriptor.to(dtype=h_base.dtype)
        cluster_confidence_tok = cluster_confidence_tok.to(dtype=h_base.dtype)
        cluster_entropy_tok = cluster_entropy_tok.to(dtype=h_base.dtype)
        
        # -----------------------------
        # Cluster semantic bridge
        # -----------------------------
        cluster_sizes = cluster_weights.sum(dim=1)  # [B, K]

        region_bridge_sum = torch.einsum(
            "btk,btd->bkd",
            cluster_weights,
            h_base,
        )  # [B, K, hidden]

        region_bridge = region_bridge_sum / (
            cluster_sizes.unsqueeze(-1).to(dtype=h_base.dtype) + 1e-6
        )  # [B, K, hidden]

        token_bridge = torch.einsum(
            "btk,bkd->btd",
            cluster_weights,
            region_bridge,
        )  # [B, T, hidden]

        bridge_delta = token_bridge - h_base  # [B, T, hidden]

        bridge_input = torch.cat(
            [
                token_bridge,
                h_base,
                bridge_delta,
                strength_tok,
                token_descriptor,
                cluster_confidence_tok,
                cluster_entropy_tok,
            ],
            dim=-1,
        )  # [B, T, 3*hidden + 1 + descriptor_dim + 2]

        bridge_signal = torch.tanh(
            self.bridge_translator(bridge_input)
        )  # [B, T, bridge_dim]

        # -----------------------------
        # Standard refine path
        # -----------------------------
        refine_input = torch.cat(
            [
                strength_tok,
                attn_context_tok,
                overlay_tok,
                bridge_signal,
            ],
            dim=-1,
        )  # [B, T, bridge_dim + 3]

        refine_hidden = torch.tanh(self.refine_in(refine_input))   # [B, T, refine_dim]
        refine_shift = self.refine_to_hidden(refine_hidden)        # [B, T, hidden]      # [B, T, hidden]


        refine_use = torch.sigmoid(
            self.refine_use_proj(refine_hidden)
        )  # [B, T, 1]

        # -----------------------------
        # Regional attractor path
        # -----------------------------

        # region summaries from h_base reference
        region_h = region_bridge  # [B, K, hidden]

        # regional overlay strength
        region_overlay_sum = torch.einsum(
            "btk,bt->bk",
            cluster_weights,
            overlay_tok.squeeze(-1),
        )  # [B, K]

        region_overlay = (
            region_overlay_sum
            / (cluster_sizes.to(dtype=h_base.dtype) + 1e-6)
        ).unsqueeze(-1)  # [B, K, 1]

        # regional attention weight
        region_attn_weight_sum = torch.einsum(
            "btk,bt->bk",
            cluster_weights,
            attn_weight_tok.squeeze(-1),
        )  # [B, K]

        region_attn_weight = (
            region_attn_weight_sum
            / (cluster_sizes.to(dtype=h_base.dtype) + 1e-6)
        ).unsqueeze(-1)  # [B, K, 1]

        # cluster presence / usage per region
        cluster_presence = cluster_weights.mean(dim=1, keepdim=False).unsqueeze(-1)  # [B, K, 1]

        # back-project regions to tokens
        token_region_h = torch.einsum(
            "btk,bkd->btd",
            cluster_weights,
            region_h,
        )  # [B, T, hidden]

        token_region_overlay = torch.einsum(
            "btk,bk->bt",
            cluster_weights,
            region_overlay.squeeze(-1),
        ).unsqueeze(-1)  # [B, T, 1]

        attractor_hidden = torch.tanh(
            self.cluster_attr_proj(token_region_h)
        )  # [B, T, hidden]

        attractor_shift = (
            token_region_overlay
            * attractor_hidden
        )  # [B, T, hidden]

        # -----------------------------
        # Token-local dual refine scale
        # -----------------------------
        tau_tok = strength_tok
        attractor_hint_tok = token_region_overlay

        dual_scale_input = torch.cat(
            [
                tau_tok,
                attractor_hint_tok,
            ],
            dim=-1,
        )  # [B, T, 2]

        dual_scale_hidden = torch.tanh(
            self.dual_refine_scale_in(dual_scale_input)
        )  # [B, T, 8]

        scale_local = torch.sigmoid(
            self.dual_refine_scale_local(dual_scale_hidden)
        )  # [B, T, 1]

        scale_attr = torch.sigmoid(
            self.dual_refine_scale_attr(dual_scale_hidden)
        )  # [B, T, 1]

        # -----------------------------
        # Base refine state — HARD GATE ABLATION TEST
        # -----------------------------
        # Temporarily bypass direct h_refine gates:
        # - refine_use
        # - scale_local
        # - scale_attr
        # - base_mix
        #
        # Keep attn_weight_tok as the attention-derived relevance mask.

        local_term = (
            attn_weight_tok.to(dtype=h_base.dtype)
            * refine_shift
        )

        attr_term = (
            self.attr_scale.to(dtype=h_base.dtype)
            * attractor_shift
        )

        # -----------------------------
        # Sequential refine hierarchy:
        # 1) Regional attractor builds macro-structure
        # 2) Local refine performs token-level correction afterward
        # -----------------------------

        # Step 1: macro structural displacement
        h_attr = h_base + attr_term

        # Step 2: local refine correction on shifted semantic terrain
        h_refine = h_attr + local_term

        # Diagnostic delta
        delta = h_refine - h_base

        # -----------------------------
        # Predictability bridge:
        # 1) build a coarse base-shifted reference from the current refine pressure
        # 2) measure refine tension relative to that shifted base
        # 3) use a shared connection-state for:
        #    - recovery of base-linked predictability
        #    - saturation tracking
        # -----------------------------
        base_shifted = h_base + 0.10 * (
            refine_use.to(dtype=h_base.dtype)
            * attn_weight_tok
            * refine_shift
        )  # [B, T, hidden]

        refine_tension = (h_refine - base_shifted).abs()  # [B, T, hidden]
        refine_tension_tok = refine_tension.mean(dim=-1, keepdim=True)  # [B, T, 1]

        connection_input = torch.cat(
            [
                h_base,
                base_shifted,
                h_refine,
                refine_tension_tok,
            ],
            dim=-1,
        )  # [B, T, 3*hidden + 1]

        connection_state = torch.tanh(
            self.connection_in(connection_input)
        )  # [B, T, hidden]

        # recovery path: bring back useful base-linked predictability information
        recovery_vec = self.recovery_proj(connection_state)  # [B, T, hidden]
        recovery_gate = torch.sigmoid(
            self.recovery_gate(connection_state)
        )  # [B, T, 1]

        recovery_term = (
            self.recovery_scale.to(dtype=h_base.dtype)
            * recovery_gate.to(dtype=h_base.dtype)
            * recovery_vec
        )  # [B, T, hidden]

        h_refine = h_refine + recovery_term

        # -----------------------------
        # Saturation vector path:
        # Saturation should actively refold distant refine structures
        # into base-compatible explanatory vectors.
        # -----------------------------
        saturation_gate = torch.sigmoid(
            self.saturation_gate(connection_state)
        )  # [B, T, 1]

        saturation_vec = self.saturation_vec_proj(connection_state)  # [B, T, hidden]

        saturation_term = (
            saturation_gate.to(dtype=h_base.dtype)
            * saturation_vec.to(dtype=h_base.dtype)
        )

        # Apply saturation as structural return flow
        h_refine = h_refine + saturation_term

        # Diagnostics
        self.last_saturation_tok = saturation_gate

        # Predictability signature now reflects:
        # "how much distant tension required structural refolding"
        predictability_signature_tok = (
            saturation_gate * refine_tension_tok
        )  # [B, T, 1]

        # expose for GPT.forward hook
        self.last_predictability_signature_tok = predictability_signature_tok

        with torch.no_grad():
            delta_h = (h_refine - h_base).abs().mean()
            self.obs_refine_use_sum += float(refine_use.mean().item())
            self.obs_delta_h_sum += float(delta_h.item())
            self.obs_scale_local_sum += float(scale_local.mean().item())
            self.obs_scale_attr_sum += float(scale_attr.mean().item())
            self.obs_recovery_gate_sum += float(recovery_gate.mean().item())
            self.obs_saturation_gate_sum += float(saturation_gate.mean().item())
            self.obs_saturation_vec_sum += float(saturation_term.abs().mean().item())
            self.obs_refine_tension_sum += float(refine_tension_tok.mean().item())
            self.obs_refine_diag_count += 1

        return self.proj(h_refine)
    
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, descriptor_dim=4 * num_kv_heads)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        prev_strength_tok: Tensor | None = None,
        pass_mode: str = "base",
    ) -> tuple[Tensor, Tensor | None]:

        if pass_mode not in ("base", "refine"):
            raise ValueError(f"invalid pass_mode: {pass_mode}")

        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # -----------------------------
        # Base pass: build reference representation
        # -----------------------------
        if pass_mode == "base":
            mlp_base_out = self.mlp.forward_base(self.mlp_norm(x))
            next_strength_tok = getattr(self.mlp, "last_strength_tok", None)

            # attention still runs to keep representation consistent
            if next_strength_tok is not None:
                self.attn.last_strength_tok = next_strength_tok
            elif prev_strength_tok is not None:
                self.attn.last_strength_tok = prev_strength_tok
            elif hasattr(self.attn, "last_strength_tok"):
                delattr(self.attn, "last_strength_tok")
            

            attn_out = self.attn(self.attn_norm(x))

            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_base_out
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

            return x, next_strength_tok

        # -----------------------------
        # Refine pass: learn structure-based shift on top of base
        # -----------------------------
        elif pass_mode == "refine":
            next_strength_tok = getattr(self.mlp, "last_strength_tok", None)

            if next_strength_tok is None:
                raise RuntimeError(
                    "refine pass requires cached base anchor, but last_strength_tok is missing. "
                    "Run base pass first."
                )

            if getattr(self.mlp, "last_h_base", None) is None:
                raise RuntimeError(
                    "refine pass requires cached base anchor, but last_h_base is missing. "
                    "Run base pass first."
                )

            self.attn.last_strength_tok = next_strength_tok

            attn_out = self.attn(self.attn_norm(x))

            # Extract attention-derived refine signals
            attn_weight_tok = getattr(self.attn, "last_role_gate_tok", None)
            attn_context_tok = getattr(self.attn, "last_role_context_tok", None)
            cluster_weights = getattr(self.attn, "last_cluster_weights", None)
            overlay_tok = getattr(self.attn, "last_overlay_tok", None)
            token_descriptor = getattr(self.attn, "last_token_descriptor", None)
            cluster_confidence_tok = getattr(self.attn, "last_cluster_confidence_tok", None)
            cluster_entropy_tok = getattr(self.attn, "last_cluster_entropy_tok", None)

            mlp_refine_out = self.mlp.forward_refine(
                attn_weight_tok,
                attn_context_tok,
                cluster_weights,
                overlay_tok,
                token_descriptor,
                cluster_confidence_tok,
                cluster_entropy_tok,
            )

            # In refine mode, do not rebuild or re-add base.
            # The residual stream already comes from the base pass; only add refine distortion.
            if mlp_refine_out is not None:
                x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_refine_out
                
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

            return x, next_strength_tok


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

        # projects token-local saturation from 1D into the model hidden space
        self.saturation_to_hidden = nn.Linear(1, model_dim, bias=False)

        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nn.init.normal_(self.saturation_to_hidden.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def set_dual_alpha(self, alpha: float) -> None:
        for block in self.blocks:
            if isinstance(block.mlp, MLP):
                block.mlp.dual_alpha = float(alpha)

    def get_predictability_signature(
        self,
        x_pred: Tensor,
        pass_mode: str = "base",
    ) -> Tensor | None:
        # Only use the signature during refine-mode prediction.
        # In base mode, keep the logits path untouched.
        if pass_mode != "refine":
            return None

        sig_sum: Tensor | None = None
        sig_count = 0

        for block in self.blocks:
            if hasattr(block, "mlp"):
                sig = getattr(block.mlp, "last_predictability_signature_tok", None)
                if sig is None:
                    continue

                # Expect shape [B, T, 1]
                sig = sig.to(dtype=x_pred.dtype)

                if sig_sum is None:
                    sig_sum = sig
                else:
                    sig_sum = sig_sum + sig

                sig_count += 1

        if sig_sum is None or sig_count == 0:
            return None

        sig_mean = sig_sum / float(sig_count)

        # keep first test bounded and smooth
        sig_mean = torch.tanh(sig_mean)

        return sig_mean

    def get_observation_stats(self) -> dict[str, float]:
        role_low_descriptor_sum = 0.0
        role_gate_sum = 0.0
        role_v_overlay_sum = 0.0
        cluster_usage_max_sum = 0.0
        cluster_usage_entropy_sum = 0.0
        cluster_presence_max_sum = 0.0
        cluster_presence_entropy_sum = 0.0
        token_overlay_mean_sum = 0.0
        region_bridge_abs_sum = 0.0
        role_count = 0.0

        refine_use_sum = 0.0
        delta_h_sum = 0.0
        scale_local_sum = 0.0
        scale_attr_sum = 0.0
        recovery_gate_sum = 0.0
        saturation_gate_sum = 0.0
        saturation_vec_sum = 0.0
        refine_tension_sum = 0.0
        refine_count = 0.0

        for block in self.blocks:
            if hasattr(block, "attn") and hasattr(block.attn, "get_role_diagnostics"):
                diag = block.attn.get_role_diagnostics()
                count = diag.get("count", 0.0)

                if count > 0:
                    role_low_descriptor_sum += diag["low_descriptor_mean"] * count
                    role_gate_sum += diag["role_gate_mean"] * count
                    role_v_overlay_sum += diag["role_v_overlay_mean"] * count
                    cluster_usage_max_sum += diag["cluster_usage_max"] * count
                    cluster_usage_entropy_sum += diag["cluster_usage_entropy"] * count
                    cluster_presence_max_sum += diag["cluster_presence_max"] * count
                    cluster_presence_entropy_sum += diag["cluster_presence_entropy"] * count
                    token_overlay_mean_sum += diag["token_overlay_mean"] * count
                    region_bridge_abs_sum += diag["region_bridge_abs_mean"] * count
                    role_count += count

            if hasattr(block, "mlp") and hasattr(block.mlp, "get_refine_diagnostics"):
                rdiag = block.mlp.get_refine_diagnostics()
                rcount = rdiag.get("count", 0.0)

                if rcount > 0:
                    refine_use_sum += rdiag["refine_use_mean"] * rcount
                    delta_h_sum += rdiag["delta_h_mean"] * rcount
                    scale_local_sum += rdiag["scale_local_mean"] * rcount
                    scale_attr_sum += rdiag["scale_attr_mean"] * rcount
                    recovery_gate_sum += rdiag["recovery_gate_mean"] * rcount
                    saturation_gate_sum += rdiag["saturation_gate_mean"] * rcount
                    saturation_vec_sum += rdiag["saturation_vec_mean"] * rcount
                    refine_tension_sum += rdiag["refine_tension_mean"] * rcount
                    refine_count += rcount

        if role_count > 0:
            role_low_descriptor_mean = role_low_descriptor_sum / role_count
            role_gate_mean = role_gate_sum / role_count
            role_v_overlay_mean = role_v_overlay_sum / role_count
            cluster_usage_max = cluster_usage_max_sum / role_count
            cluster_usage_entropy = cluster_usage_entropy_sum / role_count
            cluster_presence_max = cluster_presence_max_sum / role_count
            cluster_presence_entropy = cluster_presence_entropy_sum / role_count
            token_overlay_mean = token_overlay_mean_sum / role_count
            region_bridge_abs_mean = region_bridge_abs_sum / role_count            
        else:
            role_low_descriptor_mean = 0.0
            role_gate_mean = 0.0
            role_v_overlay_mean = 0.0
            cluster_usage_max = 0.0
            cluster_usage_entropy = 0.0
            cluster_presence_max = 0.0
            cluster_presence_entropy = 0.0
            token_overlay_mean = 0.0
            region_bridge_abs_mean = 0.0

        if refine_count > 0:
            refine_use_mean = refine_use_sum / refine_count
            delta_h_mean = delta_h_sum / refine_count
            scale_local_mean = scale_local_sum / refine_count
            scale_attr_mean = scale_attr_sum / refine_count
            recovery_gate_mean = recovery_gate_sum / refine_count
            saturation_gate_mean = saturation_gate_sum / refine_count
            saturation_vec_mean = saturation_vec_sum / refine_count
            refine_tension_mean = refine_tension_sum / refine_count
        else:
            refine_use_mean = 0.0
            delta_h_mean = 0.0
            scale_local_mean = 0.0
            scale_attr_mean = 0.0
            recovery_gate_mean = 0.0
            saturation_gate_mean = 0.0
            saturation_vec_mean = 0.0
            refine_tension_mean = 0.0

        return {
            "mean_pos": 0.0,
            "mean_neg": 0.0,
            "active_pos_frac": 0.0,
            "active_neg_frac": 0.0,
            "neg_pos_ratio": 0.0,
            "u_mean": 0.0,
            "u_std": 0.0,
            "u_pos_frac": 0.0,
            "u_neg_frac": 0.0,
            "pos_energy_ratio": 0.0,
            "neg_energy_ratio": 0.0,
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "alpha_tok_mean": 0.0,
            "alpha_tok_std": 0.0,
            "tau_tok_mean": 0.0,
            "tau_tok_std": 0.0,
            "base_keep_tok_mean": 0.0,
            "base_keep_tok_std": 0.0,
            "delta_gate_scale": 0.0,
            "role_low_descriptor_mean": role_low_descriptor_mean,
            "role_gate_mean": role_gate_mean,
            "role_v_overlay_mean": role_v_overlay_mean,
            "cluster_usage_max": cluster_usage_max,
            "cluster_usage_entropy": cluster_usage_entropy,
            "cluster_presence_max": cluster_presence_max,
            "cluster_presence_entropy": cluster_presence_entropy,
            "token_overlay_mean": token_overlay_mean,
            "region_bridge_abs_mean": region_bridge_abs_mean,
            "refine_use_mean": refine_use_mean,
            "delta_h_mean": delta_h_mean,
            "scale_local_mean": scale_local_mean,
            "scale_attr_mean": scale_attr_mean,
            "recovery_gate_mean": recovery_gate_mean,
            "saturation_gate_mean": saturation_gate_mean,
            "saturation_vec_mean": saturation_vec_mean,
            "refine_tension_mean": refine_tension_mean,
            "elem_count": 0.0,
        }
    
    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        pass_mode: str = "base",
    ) -> Tensor:
        if pass_mode not in ("base", "refine"):
            raise ValueError(f"invalid pass_mode: {pass_mode}")

        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[tuple[Tensor, Tensor | None]] = []

        prev_strength_tok: Tensor | None = None

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x, prev_strength_tok = self.blocks[i](x, x0, prev_strength_tok, pass_mode=pass_mode)
            skips.append((x, prev_strength_tok))

        for i in range(self.num_decoder_layers):
            if skips:
                skip_x, skip_strength_tok = skips.pop()
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip_x

                if skip_strength_tok is not None:
                    if prev_strength_tok is None:
                        prev_strength_tok = skip_strength_tok
                    else:
                        prev_strength_tok = 0.5 * (prev_strength_tok + skip_strength_tok)

            x, prev_strength_tok = self.blocks[
                self.num_encoder_layers + i
            ](x, x0, prev_strength_tok, pass_mode=pass_mode)

        x = self.final_norm(x)

        # final hidden state used for token prediction
        x_pred = x
        targets = target_ids.reshape(-1)

        # optional hidden-space saturation hook
        saturation_tok = None
        if hasattr(self, "get_predictability_signature"):
            saturation_tok = self.get_predictability_signature(x_pred, pass_mode=pass_mode)

        if saturation_tok is not None:
            # project 1D saturation into a learned hidden-space steering signal
            sat_signal = self.saturation_to_hidden(
                saturation_tok.to(dtype=x_pred.dtype)
            )
            x_pred = x_pred + 0.45 * sat_signal

        x_flat = x_pred.reshape(-1, x_pred.size(-1))

        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

        loss_per_tok = F.cross_entropy(logits.float(), targets, reduction="none")
        loss_plain = loss_per_tok.mean()

        return loss_plain
    
# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = zeropower_via_newtonschulz5
    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model


    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=True")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def grad_norm_group(model, name_parts):
        total = 0.0
        max_norm = 0.0
        count = 0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if any(part in name for part in name_parts):
                g = p.grad.detach()
                n = float(g.float().norm().item())
                total += n
                max_norm = max(max_norm, n)
                count += 1
        mean_norm = total / max(count, 1)
        return mean_norm, max_norm, count
    
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        grad_stats = None
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            x, y = train_loader.next_batch(
                args.train_batch_tokens,
                args.train_seq_len,
                grad_accum_steps,
            )

            # -----------------------------
            # Pass 1: base only prepares the first internal view
            # no parameter-space transfer from base
            # -----------------------------
            zero_grad_all()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _ = model(x, y, pass_mode="base")

            # -----------------------------
            # Pass 2: refine performs the actual transfer into parameter space
            # -----------------------------
            zero_grad_all()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss_refine = model(x, y, pass_mode="refine")
            (loss_refine * grad_scale).backward()

            # --- Gradient diagnostics (aggregated) ---
            if micro_step == grad_accum_steps - 1:
                grad_stats = {}
                for key, parts in {
                    "grad_refine": ["refine_in", "refine_to_hidden"],
                    "grad_refine_use": ["refine_use_proj"],
                    "grad_scale": ["dual_refine_scale"],
                    "grad_attractor": ["cluster_attr_proj"],
                    "grad_recovery": ["recovery_proj", "recovery_gate"],
                    "grad_saturation": ["saturation_gate", "saturation_vec_proj"],
                    "grad_role_gate": ["role_gate_proj"],
                    "grad_overlay": ["descriptor_to_k", "descriptor_to_v"],
                }.items():
                    mean_norm, max_norm, count = grad_norm_group(base_model, parts)
                    grad_stats[key] = mean_norm
                    grad_stats[f"{key}_max"] = max_norm
                    grad_stats[f"{key}_count"] = count

            train_loss += loss_refine.detach()


        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()


        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        
        if should_log_train:
            obs_stats = (
                base_model.get_observation_stats()
                if hasattr(base_model, "get_observation_stats")
                else {
                    "role_low_descriptor_mean": 0.0,
                    "role_gate_mean": 0.0,
                    "role_v_overlay_mean": 0.0,
                    "cluster_usage_max": 0.0,
                    "cluster_usage_entropy": 0.0,
                    "cluster_presence_max": 0.0,
                    "cluster_presence_entropy": 0.0,
                    "token_overlay_mean": 0.0,
                    "region_bridge_abs_mean": 0.0,
                    "refine_use_mean": 0.0,
                    "delta_h_mean": 0.0,
                    "recovery_gate_mean": 0.0,
                    "saturation_gate_mean": 0.0,
                    "saturation_vec_mean": 0.0,
                    "refine_tension_mean": 0.0,
                    "scale_local_mean": 0.0,
                    "scale_attr_mean": 0.0

                }
            )
            grad_str = ""
            if grad_stats is not None:
                grad_str = (
                    f"grad_refine:{grad_stats['grad_refine']:.3e} "
                    f"grad_refine_max:{grad_stats['grad_refine_max']:.3e} "
                    f"grad_refine_count:{grad_stats['grad_refine_count']} "
                    f"grad_refine_use:{grad_stats['grad_refine_use']:.3e} "
                    f"grad_refine_use_max:{grad_stats['grad_refine_use_max']:.3e} "
                    f"grad_refine_use_count:{grad_stats['grad_refine_use_count']} "
                    f"grad_scale:{grad_stats['grad_scale']:.3e} "
                    f"grad_attractor:{grad_stats['grad_attractor']:.3e} "
                    f"grad_recovery:{grad_stats['grad_recovery']:.3e} "
                    f"grad_saturation:{grad_stats['grad_saturation']:.3e} "
                    f"grad_role_gate:{grad_stats['grad_role_gate']:.3e} "
                    f"grad_role_gate_max:{grad_stats['grad_role_gate_max']:.3e} "
                    f"grad_role_gate_count:{grad_stats['grad_role_gate_count']} "
                    f"grad_overlay:{grad_stats['grad_overlay']:.3e} "
                )
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"role_low_descriptor_mean:{obs_stats['role_low_descriptor_mean']:.6f} "
                f"role_gate_mean:{obs_stats['role_gate_mean']:.6f} "
                f"role_v_overlay_mean:{obs_stats['role_v_overlay_mean']:.6f} "
                f"cluster_usage_max:{obs_stats['cluster_usage_max']:.6f} "
                f"cluster_usage_entropy:{obs_stats['cluster_usage_entropy']:.6f} "
                f"cluster_presence_max:{obs_stats['cluster_presence_max']:.6f} "
                f"cluster_presence_entropy:{obs_stats['cluster_presence_entropy']:.6f} "
                f"token_overlay_mean:{obs_stats['token_overlay_mean']:.6f} "
                f"region_bridge_abs_mean:{obs_stats['region_bridge_abs_mean']:.6f} "
                f"refine_use_mean:{obs_stats['refine_use_mean']:.6f} "
                f"delta_h_mean:{obs_stats['delta_h_mean']:.6f} "
                f"recovery_gate_mean:{obs_stats['recovery_gate_mean']:.6f} "
                f"saturation_gate_mean:{obs_stats['saturation_gate_mean']:.6f} "
                f"saturation_vec_mean:{obs_stats['saturation_vec_mean']:.6f} "
                f"refine_tension_mean:{obs_stats['refine_tension_mean']:.6f} "
                f"scale_local_mean:{obs_stats['scale_local_mean']:.6f} "
                f"scale_attr_mean:{obs_stats['scale_attr_mean']:.6f} "
                f"{grad_str}"
            )

            if hasattr(base_model, "blocks"):
                for block in base_model.blocks:
                    if hasattr(block, "attn") and hasattr(block.attn, "reset_role_diagnostics"):
                        block.attn.reset_role_diagnostics()
                    if hasattr(block, "mlp") and hasattr(block.mlp, "reset_refine_diagnostics"):
                        block.mlp.reset_refine_diagnostics()

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
