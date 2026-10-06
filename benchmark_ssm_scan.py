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
import json
import sys
import time
from pathlib import Path

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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this benchmark requires the target GPU")
    if args.backend == "triton" and triton is None:
        parser.error("Triton is not installed in this Python environment")
    if args.seconds <= 0 or args.warmup < 0 or args.progress_interval <= 0:
        parser.error("seconds and progress interval must be positive; warmup must be nonnegative")
    if args.compile_mode != "none" and args.warmup == 0:
        parser.error("compiled runs need at least one warmup step to exclude compilation from timing")

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
    for layer in model.modules():
        if hasattr(layer, "scan_backend"):
            layer.scan_backend = args.backend
    if args.compile_mode != "none":
        model = torch.compile(model, mode=args.compile_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    if args.scope == "ssm":
        x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)
        y = None
    else:
        x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model(x)
            if args.scope == "ssm":
                loss = result.square().mean()
            else:
                loss = F.cross_entropy(result.reshape(-1, args.vocab_size), y.reshape(-1))
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
    next_progress = args.progress_interval
    while time.perf_counter() - start < args.seconds:
        loss = step()
        steps += 1
        elapsed_now = time.perf_counter() - start
        if elapsed_now >= next_progress:
            print(f"progress {args.model}/{args.backend}/{args.compile_mode}: {elapsed_now:.1f}s, "
                  f"{steps} steps, {steps * args.batch_size * args.seq_len / elapsed_now:.1f} tok/s",
                  flush=True)
            next_progress += args.progress_interval
    elapsed = time.perf_counter() - start
    tokens = steps * args.batch_size * args.seq_len
    print(json.dumps({
        "model": args.model,
        "scope": args.scope,
        "backend": args.backend,
        "compile_mode": args.compile_mode,
        "warmup_seconds": round(warmup_seconds, 3),
        "seconds_requested": args.seconds,
        "seconds_elapsed": round(elapsed, 3),
        "steps": steps,
        "tokens_per_second": round(tokens / elapsed, 2),
        "seconds_per_step": round(elapsed / steps, 5),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "last_loss": loss,
        "shape": [args.batch_size, args.seq_len, args.dim],
    }, indent=2))


if __name__ == "__main__":
    main()
