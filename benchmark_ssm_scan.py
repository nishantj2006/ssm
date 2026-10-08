"""Time a pure or hybrid training workload on dense and Triton SSM backends.

Run each backend as a separate process, for example:
  python benchmark_ssm_scan.py --model pure --backend dense --seconds 600
  python benchmark_ssm_scan.py --model pure --backend triton --seconds 600
  python benchmark_ssm_scan.py --model hybrid --backend dense --seconds 600
  python benchmark_ssm_scan.py --model hybrid --backend triton --seconds 600
  python benchmark_ssm_scan.py --model pure --backend dense --compile-mode default --seconds 180
  python benchmark_ssm_scan.py --model pure --backend triton --compile-mode default --seconds 180
The timer starts after warmup and excludes checkpoint loading and compilation.
"""

import argparse
from collections import deque
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model" / "pure"))
sys.path.insert(0, str(ROOT / "model" / "hybrid"))
from single_ssm import PureSSMLanguageModel, SingleHeadSSMLayer as PureSSMLayer
from ssm_full import HybridLanguageModel, SingleHeadSSMLayer as HybridSSMLayer
from ssm_scan import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pure", "hybrid"), required=True)
    parser.add_argument("--backend", choices=("dense", "triton"), required=True)
    parser.add_argument("--optimizer", choices=("adamw", "adafactor"), default="adamw")
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--fused-loss", action="store_true",
                        help="use Liger fused linear cross-entropy for the pure model")
    parser.add_argument("--compile-mode", choices=("none", "default", "max-autotune"), default="none")
    parser.add_argument("--scope", choices=("model", "ssm"), default="model")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--progress-interval", type=float, default=60)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--data-path", type=Path,
                        help="uint16 token file; use sequential, nonrepeating windows")
    parser.add_argument("--save-checkpoint", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this benchmark requires the target GPU")
    if args.backend == "triton" and triton is None:
        parser.error("Triton is not installed in this Python environment")
    if args.seconds <= 0 or args.warmup < 0 or args.progress_interval <= 0:
        parser.error("seconds and progress interval must be positive; warmup must be nonnegative")
    if args.compile_mode != "none" and args.warmup == 0:
        parser.error("compiled runs need at least one warmup step to exclude compilation from timing")
    if args.data_path and args.scope != "model":
        parser.error("--data-path requires --scope model")
    if args.fused_loss and (args.model != "pure" or args.scope != "model"):
        parser.error("--fused-loss requires --model pure --scope model")

    fused_loss = None
    if args.fused_loss:
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        except ImportError as exc:
            parser.error(f"--fused-loss requires liger-kernel: {exc}")
        fused_loss = LigerFusedLinearCrossEntropyLoss()

    torch.manual_seed(42)
    device = "cuda"
    if args.scope == "ssm":
        model = (PureSSMLayer if args.model == "pure" else HybridSSMLayer)(args.dim)
    elif args.model == "pure":
        model = PureSSMLanguageModel(args.vocab_size, args.dim, args.layers)
    else:
        model = HybridLanguageModel(args.vocab_size, args.dim, args.layers,
                                    attn_layers=(2, 5))
    model = model.to(device).train()
    if args.checkpoint_blocks and args.scope == "model":
        model.gradient_checkpointing = True
    for layer in model.modules():
        if hasattr(layer, "scan_backend"):
            layer.scan_backend = args.backend
    if args.compile_mode != "none":
        model = torch.compile(model, mode=args.compile_mode)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    else:
        optimizer = torch.optim.Adafactor(model.parameters(), lr=3e-4)
    data = None
    data_cursor = 0
    if args.data_path:
        data = np.memmap(args.data_path, dtype=np.uint16, mode="r")
        if len(data) < (args.warmup + 1) * args.batch_size * args.seq_len + 1:
            parser.error("token file is too small for warmup and training")
        if int(data.max()) >= args.vocab_size:
            parser.error("token file has IDs outside the model vocabulary")
    if args.scope == "ssm":
        x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)
        y = None
    elif data is None:
        x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)

    def next_batch():
        nonlocal data_cursor
        count = args.batch_size * args.seq_len
        start = data_cursor * count
        if start + count >= len(data):
            raise RuntimeError("Prepared token file exhausted; prepare more FineWeb tokens")
        segment = np.asarray(data[start:start + count + 1], dtype=np.int64)
        data_cursor += 1
        inputs = torch.from_numpy(segment[:-1].copy()).reshape(args.batch_size, args.seq_len).to(device)
        targets = torch.from_numpy(segment[1:].copy()).reshape(args.batch_size, args.seq_len).to(device)
        return inputs, targets

    def step():
        batch_x, batch_y = next_batch() if data is not None else (x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if fused_loss is not None:
                hidden = model.forward_hidden(batch_x)
                loss = fused_loss(model.classifier.weight, hidden.reshape(-1, args.dim),
                                  batch_y.reshape(-1))
            else:
                result = model(batch_x)
            if args.scope == "ssm":
                loss = result.square().mean()
            elif fused_loss is None:
                loss = F.cross_entropy(result.reshape(-1, args.vocab_size), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return loss.item()

    warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    warmup_seconds = time.perf_counter() - warmup_start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    steps = 0
    loss = float("nan")
    first_losses = []
    recent_losses = deque(maxlen=20)
    next_progress = args.progress_interval
    while time.perf_counter() - start < args.seconds:
        loss = step()
        steps += 1
        if len(first_losses) < 20:
            first_losses.append(loss)
        recent_losses.append(loss)
        elapsed_now = time.perf_counter() - start
        if elapsed_now >= next_progress:
            print(f"progress {args.model}/{args.backend}/{args.compile_mode}: {elapsed_now:.1f}s, "
                  f"{steps} steps, {steps * args.batch_size * args.seq_len / elapsed_now:.1f} tok/s",
                  flush=True)
            next_progress += args.progress_interval
    elapsed = time.perf_counter() - start
    tokens = steps * args.batch_size * args.seq_len
    if args.save_checkpoint:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_name": args.optimizer,
                    "training_tokens": tokens,
                    "data_cursor": data_cursor,
                    "data_path": str(args.data_path) if args.data_path else None},
                   args.save_checkpoint)
    print(json.dumps({
        "model": args.model,
        "scope": args.scope,
        "backend": args.backend,
        "optimizer": args.optimizer,
        "checkpoint_blocks": args.checkpoint_blocks,
        "fused_loss": args.fused_loss,
        "compile_mode": args.compile_mode,
        "warmup_seconds": round(warmup_seconds, 3),
        "seconds_requested": args.seconds,
        "seconds_elapsed": round(elapsed, 3),
        "steps": steps,
        "data_path": str(args.data_path) if args.data_path else None,
        "train_tokens": tokens,
        "data_cursor": data_cursor if data is not None else None,
        "checkpoint_path": str(args.save_checkpoint) if args.save_checkpoint else None,
        "tokens_per_second": round(tokens / elapsed, 2),
        "seconds_per_step": round(elapsed / steps, 5),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "last_loss": loss,
        "mean_first_20_losses": round(sum(first_losses) / len(first_losses), 4),
        "mean_last_20_losses": round(sum(recent_losses) / len(recent_losses), 4),
        "shape": [args.batch_size, args.seq_len, args.dim],
    }, indent=2))


if __name__ == "__main__":
    main()
