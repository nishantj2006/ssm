"""Train a pure SSM over a cleaned FineWeb token stream.

Pilot initialization requires the full training file to extend the pilot's
token stream. Fresh initialization starts from random weights. Each pass uses
nonoverlapping, sequential token windows.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_scan import triton
from single_ssm import PureSSMLanguageModel


VOCAB_SIZE = 50304
DIM = 512
LAYERS = 8


def verify_prefix(full_path, prefix_path):
    """Reject a corpus whose initial tokens differ from the pilot corpus."""
    if full_path.stat().st_size < prefix_path.stat().st_size:
        raise ValueError("Full corpus is shorter than the pilot corpus")
    with full_path.open("rb") as full, prefix_path.open("rb") as prefix:
        while chunk := prefix.read(1 << 20):
            if full.read(len(chunk)) != chunk:
                raise ValueError("Full corpus differs from the pilot token stream")


def read_manifest(data_path):
    manifest = json.loads((data_path.parent / "manifest.json").read_text())
    if manifest["vocab_size"] != VOCAB_SIZE or manifest["max_train_token_id"] >= VOCAB_SIZE:
        raise ValueError("Prepared corpus has incompatible token IDs")
    if data_path.stat().st_size != 2 * manifest["train_tokens"]:
        raise ValueError("Prepared corpus size does not match its manifest")
    return manifest


def save_checkpoint(path, model, optimizer, config, pass_index, step_in_pass,
                    completed_updates, schedule_start_update, schedule_total_updates,
                    train_tokens, last_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_name": "adamw",
        "fused_loss": True,
        "training_config": config,
        "pass_index": pass_index,
        "step_in_pass": step_in_pass,
        "completed_updates": completed_updates,
        "schedule_start_update": schedule_start_update,
        "schedule_total_updates": schedule_total_updates,
        "training_tokens": train_tokens,
        "last_loss": last_loss,
    }, temporary)
    os.replace(temporary, path)


def validate(model, tokens, count=100):
    model.eval()
    window = 4 * 512
    available = (len(tokens) - 1) // window
    starts = np.linspace(0, (available - 1) * window,
                         min(count, available), dtype=np.int64)
    losses = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in starts:
            sample = np.asarray(tokens[start:start + window + 1], dtype=np.int64)
            x = torch.from_numpy(sample[:-1].copy()).reshape(4, 512).cuda()
            y = torch.from_numpy(sample[1:].copy()).reshape(4, 512).cuda()
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses), len(losses) * window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--expected-prefix", type=Path)
    parser.add_argument("--pilot-checkpoint", type=Path)
    parser.add_argument("--init", choices=("pilot", "fresh"), default="pilot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--final-lr", type=float, default=3e-5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--max-additional-updates", type=int, default=None,
                        help="Stop early after this many updates; useful for a smoke test")
    args = parser.parse_args()

    if not torch.cuda.is_available() or triton is None:
        parser.error("This full run requires CUDA and Triton")
    if min(args.passes, args.batch_size, args.seq_len, args.dim, args.layers) < 1:
        parser.error("passes, batch size, sequence length, dim, and layers must be positive")
    if args.log_every < 1 or args.validate_every < 1 or args.checkpoint_every < 1:
        parser.error("logging, validation, and checkpoint intervals must be positive")
    if not (0 < args.final_lr <= args.peak_lr):
        parser.error("learning rates must satisfy 0 < final <= peak")

    manifest = read_manifest(args.data_path)
    if args.init == "pilot":
        if args.expected_prefix is None or args.pilot_checkpoint is None:
            parser.error("pilot initialization requires --expected-prefix and --pilot-checkpoint")
        if args.dim != DIM or args.layers != LAYERS:
            parser.error("pilot checkpoint requires dim=512 and layers=8")
        verify_prefix(args.data_path, args.expected_prefix)
    data = np.memmap(args.data_path, dtype=np.uint16, mode="r")
    validation = np.memmap(args.validation_path, dtype=np.uint16, mode="r")
    if len(validation) < 4 * 512 + 1:
        parser.error("Validation token file is too short")
    tokens_per_update = args.batch_size * args.seq_len
    steps_per_pass = (len(data) - 1) // tokens_per_update
    schedule_total_updates = steps_per_pass * args.passes
    if steps_per_pass < 1:
        parser.error("Training token file is too short")

    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    fused_loss = LigerFusedLinearCrossEntropyLoss()
    model = PureSSMLanguageModel(VOCAB_SIZE, args.dim, args.layers).cuda().train()
    for layer in model.modules():
        if hasattr(layer, "scan_backend"):
            layer.scan_backend = "triton"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, fused=True)

    latest = args.output_dir / "latest_checkpoint.pt"
    config = {"dim": args.dim, "layers": args.layers,
              "batch_size": args.batch_size, "seq_len": args.seq_len}
    resume_path = latest if latest.exists() else (
        args.pilot_checkpoint if args.init == "pilot" else None)
    checkpoint = (torch.load(resume_path, map_location="cpu", weights_only=False)
                  if resume_path is not None else None)
    if checkpoint is not None:
        if resume_path == latest:
            saved_config = checkpoint.get("training_config", {
                "dim": DIM, "layers": LAYERS, "batch_size": 32, "seq_len": 512})
            if saved_config != config:
                raise ValueError("Checkpoint model or batch configuration differs from this run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if resume_path == args.pilot_checkpoint and args.init == "pilot":
        if args.batch_size != 32 or args.seq_len != 512:
            parser.error("The pilot checkpoint uses batch 32 and sequence 512")
        pass_index = 0
        step_in_pass = int(checkpoint["data_cursor"])
        completed_updates = step_in_pass
        schedule_start_update = completed_updates
        train_tokens = completed_updates * tokens_per_update
    elif checkpoint is not None:
        pass_index = int(checkpoint["pass_index"])
        step_in_pass = int(checkpoint["step_in_pass"])
        completed_updates = int(checkpoint["completed_updates"])
        schedule_start_update = int(checkpoint["schedule_start_update"])
        train_tokens = int(checkpoint["training_tokens"])
        if checkpoint["schedule_total_updates"] != schedule_total_updates:
            raise ValueError("Corpus size, pass count, or batch size changed since checkpoint")
    else:
        pass_index = 0
        step_in_pass = 0
        completed_updates = 0
        schedule_start_update = 0
        train_tokens = 0
    if step_in_pass > steps_per_pass or pass_index > args.passes:
        raise ValueError("Checkpoint cursor is outside the prepared corpus")
    if completed_updates != pass_index * steps_per_pass + step_in_pass:
        raise ValueError("Checkpoint update count does not match its data cursor")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    start = time.perf_counter()
    interval_start = start
    interval_updates = 0
    interval_loss = 0.0
    added_updates = 0
    last_loss = float("nan")
    print(json.dumps({
        "event": "start", "resume_path": str(resume_path) if resume_path else None,
        "source": manifest["source"], "train_tokens": len(data),
        "training_config": config,
        "steps_per_pass": steps_per_pass, "passes": args.passes,
        "completed_updates": completed_updates,
        "schedule_total_updates": schedule_total_updates,
    }), flush=True)

    with metrics_path.open("a", buffering=1) as metrics:
        while pass_index < args.passes:
            while step_in_pass < steps_per_pass:
                start_token = step_in_pass * tokens_per_update
                sample = np.asarray(data[start_token:start_token + tokens_per_update + 1],
                                    dtype=np.int64)
                x = torch.from_numpy(sample[:-1].copy()).reshape(args.batch_size,
                                                                   args.seq_len).cuda()
                y = torch.from_numpy(sample[1:].copy()).reshape(args.batch_size,
                                                                 args.seq_len).cuda()
                progress = (completed_updates - schedule_start_update) / max(
                    1, schedule_total_updates - schedule_start_update)
                learning_rate = args.final_lr + (args.peak_lr - args.final_lr) * (
                    1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.forward_hidden(x)
                    loss = fused_loss(model.classifier.weight,
                                      hidden.reshape(-1, args.dim), y.reshape(-1))
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                last_loss = loss.item()
                if not math.isfinite(last_loss) or not torch.isfinite(gradient_norm).item():
                    raise FloatingPointError("Nonfinite loss or gradient norm; checkpoint retained")
                optimizer.step()

                step_in_pass += 1
                completed_updates += 1
                added_updates += 1
                train_tokens += tokens_per_update
                interval_updates += 1
                interval_loss += last_loss

                if completed_updates % args.log_every == 0:
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    record = {
                        "event": "progress", "pass": pass_index + 1,
                        "completed_updates": completed_updates,
                        "step_in_pass": step_in_pass,
                        "training_tokens": train_tokens,
                        "loss": round(interval_loss / interval_updates, 5),
                        "lr": learning_rate,
                        "tokens_per_second": round(
                            interval_updates * tokens_per_update / (now - interval_start), 2),
                        "elapsed_hours": round((now - start) / 3600, 3),
                    }
                    print(json.dumps(record), flush=True)
                    metrics.write(json.dumps(record) + "\n")
                    interval_start = now
                    interval_updates = 0
                    interval_loss = 0.0

                if completed_updates % args.validate_every == 0:
                    valid_loss, valid_tokens = validate(model, validation)
                    record = {"event": "validation", "completed_updates": completed_updates,
                              "validation_tokens": valid_tokens,
                              "validation_loss": round(valid_loss, 5)}
                    print(json.dumps(record), flush=True)
                    metrics.write(json.dumps(record) + "\n")

                if completed_updates % args.checkpoint_every == 0:
                    save_checkpoint(latest, model, optimizer, config, pass_index, step_in_pass,
                                    completed_updates, schedule_start_update,
                                    schedule_total_updates, train_tokens, last_loss)
                    print(json.dumps({"event": "checkpoint",
                                      "completed_updates": completed_updates,
                                      "path": str(latest)}), flush=True)

                if (args.max_additional_updates is not None
                        and added_updates >= args.max_additional_updates):
                    save_checkpoint(latest, model, optimizer, config, pass_index, step_in_pass,
                                    completed_updates, schedule_start_update,
                                    schedule_total_updates, train_tokens, last_loss)
                    print(json.dumps({"event": "stopped_after_smoke_test",
                                      "completed_updates": completed_updates}), flush=True)
                    return

            pass_index += 1
            step_in_pass = 0
            save_checkpoint(latest, model, optimizer, config, pass_index, step_in_pass,
                            completed_updates, schedule_start_update,
                            schedule_total_updates, train_tokens, last_loss)
            print(json.dumps({"event": "pass_complete", "passes_complete": pass_index,
                              "completed_updates": completed_updates}), flush=True)

    valid_loss, valid_tokens = validate(model, validation)
    print(json.dumps({"event": "complete", "passes": args.passes,
                      "completed_updates": completed_updates,
                      "training_tokens": train_tokens,
                      "validation_tokens": valid_tokens,
                      "validation_loss": valid_loss,
                      "elapsed_hours": (time.perf_counter() - start) / 3600}), flush=True)


if __name__ == "__main__":
    main()
