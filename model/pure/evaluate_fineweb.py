"""Score the final pure SSM checkpoint on the full held-out FineWeb split."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_scan import triton
from single_ssm import PureSSMLanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available() or triton is None:
        parser.error("Evaluation requires CUDA and Triton")
    if args.batch_size < 1 or args.seq_len < 1:
        parser.error("Batch size and sequence length must be positive")
    tokens = np.memmap(args.validation_path, dtype=np.uint16, mode="r")
    window = args.batch_size * args.seq_len
    batches = (len(tokens) - 1) // window
    if batches < 1:
        parser.error("Validation split is too short")
    if int(tokens.max()) >= 50304:
        parser.error("Validation split contains an out-of-vocabulary token")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("training_config", {"dim": 512, "layers": 8})
    model = PureSSMLanguageModel(50304, config["dim"], config["layers"]).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    for layer in model.modules():
        if hasattr(layer, "scan_backend"):
            layer.scan_backend = "triton"

    weighted_loss = 0.0
    correct = 0
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in range(batches):
            start = batch * window
            sample = np.asarray(tokens[start:start + window + 1], dtype=np.int64)
            x = torch.from_numpy(sample[:-1].copy()).reshape(args.batch_size,
                                                               args.seq_len).cuda()
            y = torch.from_numpy(sample[1:].copy()).reshape(args.batch_size,
                                                             args.seq_len).cuda()
            logits = model(x).reshape(-1, 50304)
            labels = y.reshape(-1)
            weighted_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            if (batch + 1) % 1000 == 0:
                print(json.dumps({"event": "evaluation_progress", "batches": batch + 1,
                                  "total_batches": batches}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scored_tokens = batches * window
    mean_loss = weighted_loss / scored_tokens
    result = {
        "checkpoint": str(args.checkpoint),
        "validation_path": str(args.validation_path),
        "validation_tokens": scored_tokens,
        "cross_entropy_nats": mean_loss,
        "bits_per_token": mean_loss / math.log(2),
        "perplexity": math.exp(mean_loss),
        "top1_accuracy": correct / scored_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": scored_tokens / elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "evaluation_complete", **result}), flush=True)


if __name__ == "__main__":
    main()
