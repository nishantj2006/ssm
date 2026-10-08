"""Stream a cleaned, GPT-2-tokenized portion of FineWeb sample-10BT.

The full source contains about 10 billion GPT-2 tokens. This script can
prepare a bounded subset for a timed training run without downloading the
whole 27.6 GB Parquet sample first.
"""

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


SOURCE = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
EOT = 50256
VOCAB_SIZE = 50304
SPECIAL_MARKER = re.compile(r"(?i)<\|[a-z_]+\|>|<unk>|\[unk\]|<unknown>")
HEADING = re.compile(r"^\s*(?:#{1,6}|={1,6})\s*(.*?)\s*(?:#{1,6}|={1,6})?\s*$")
BOILERPLATE = re.compile(
    r"(?i)^\s*(?:skip to (?:main )?content|accept (?:all )?cookies|"
    r"cookie (?:preferences|settings)|sign in|log in|register|subscribe|"
    r"enable javascript|all rights reserved|privacy policy|terms of (?:use|service))\s*$"
)
META_LINE = re.compile(
    r"(?i)^(?:category archives?:|viewing single post|posted on |"
    r"related posts|recent posts|leave a reply|share this|read more(?:\.\.\.)?$)"
)


def clean_document(text, language_score, minimum_chars=400):
    """Return readable prose or None for short, noisy, or mostly tabular pages."""
    if not isinstance(text, str) or language_score < 0.90 or len(text) < minimum_chars:
        return None
    text = unicodedata.normalize("NFC", text)
    if text.count("\ufffd") > 1 or text.count("|") > len(text) * 0.02:
        return None
    text = SPECIAL_MARKER.sub(" ", text)
    text = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\n\t" else ch
                   for ch in text)
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip(" |")
        if not line or BOILERPLATE.fullmatch(line) or META_LINE.match(line):
            continue
        if line.count("|") >= 2:
            continue
        line = line.replace("|", " ")
        match = HEADING.fullmatch(line)
        if match:
            line = match.group(1).strip()
        if line:
            lines.append(line)
    if not lines:
        return None
    text = "\n".join(lines)
    visible = [ch for ch in text if not ch.isspace()]
    if len(text) < minimum_chars or not visible:
        return None
    if sum(ch.isalnum() for ch in visible) / len(visible) < 0.60:
        return None
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.65:
        return None
    return text


def validation_document(identifier):
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100 == 0


def prepare(output_dir, target_train_tokens, progress_interval):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    valid_path = output_dir / "validation.bin"
    if train_path.exists() or valid_path.exists():
        raise FileExistsError("Output already exists; choose an empty directory")

    train_partial = train_path.with_suffix(".bin.partial")
    valid_partial = valid_path.with_suffix(".bin.partial")
    state_path = output_dir / "prepare_state.json"
    if train_partial.exists() or valid_partial.exists() or state_path.exists():
        if not (train_partial.exists() and valid_partial.exists() and state_path.exists()):
            raise RuntimeError("Incomplete partial corpus without resume state")
        state = json.loads(state_path.read_text())
        counters = Counter(state["counters"])
        train_bytes = int(state["train_bytes"])
        valid_bytes = int(state["valid_bytes"])
        if train_partial.stat().st_size < train_bytes or valid_partial.stat().st_size < valid_bytes:
            raise RuntimeError("Partial corpus is shorter than its saved resume state")
        with train_partial.open("r+b") as train_file:
            train_file.truncate(train_bytes)
        with valid_partial.open("r+b") as valid_file:
            valid_file.truncate(valid_bytes)
        prior_elapsed = float(state["elapsed_seconds"])
        print(json.dumps({"event": "resume", "documents_seen": counters["documents_seen"],
                          "train_tokens": counters["train_tokens"]}), flush=True)
    else:
        counters = Counter()
        prior_elapsed = 0.0
        train_partial.touch()
        valid_partial.touch()
        state_path.write_text(json.dumps({"counters": {}, "train_bytes": 0,
                                          "valid_bytes": 0, "elapsed_seconds": 0}) + "\n")

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(SOURCE, CONFIG, split="train", streaming=True)
    records = iter(dataset)
    start = time.monotonic()
    next_progress = progress_interval
    try:
        for _ in range(counters["documents_seen"]):
            next(records)
        with train_partial.open("ab") as train_file, valid_partial.open("ab") as valid_file:
            for row in records:
                counters["documents_seen"] += 1
                cleaned = clean_document(row.get("text"), row.get("language_score") or 0)
                if cleaned is None:
                    counters["documents_rejected"] += 1
                    continue
                ids = tokenizer.encode_ordinary(cleaned)
                if len(ids) < 80 or len(ids) > 8192:
                    counters["documents_rejected_length"] += 1
                    continue
                if not ids or min(ids) < 0 or max(ids) >= EOT:
                    raise ValueError("Unexpected GPT-2 token ID in cleaned text")
                ids.append(EOT)
                packed = np.asarray(ids, dtype=np.uint16)
                if validation_document(str(row.get("id", ""))):
                    valid_file.write(packed.tobytes())
                    counters["validation_documents"] += 1
                    counters["validation_tokens"] += len(ids)
                else:
                    train_file.write(packed.tobytes())
                    counters["train_documents"] += 1
                    counters["train_tokens"] += len(ids)
                elapsed = time.monotonic() - start
                if elapsed >= next_progress:
                    train_file.flush()
                    valid_file.flush()
                    os.fsync(train_file.fileno())
                    os.fsync(valid_file.fileno())
                    state = {"counters": dict(counters),
                             "train_bytes": train_file.tell(),
                             "valid_bytes": valid_file.tell(),
                             "elapsed_seconds": prior_elapsed + elapsed}
                    temporary_state = state_path.with_suffix(".json.tmp")
                    temporary_state.write_text(json.dumps(state) + "\n")
                    temporary_state.replace(state_path)
                    print(json.dumps({"elapsed_seconds": round(prior_elapsed + elapsed, 1),
                                      **counters}), flush=True)
                    next_progress += progress_interval
                if target_train_tokens is not None and counters["train_tokens"] >= target_train_tokens:
                    break
        if (target_train_tokens is not None and counters["train_tokens"] < target_train_tokens
                or counters["validation_tokens"] < 1000):
            raise RuntimeError("Source ended before enough train and validation tokens were prepared")
        train_partial.replace(train_path)
        valid_partial.replace(valid_path)
    finally:
        records.close()

    counts = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid = np.memmap(valid_path, dtype=np.uint16, mode="r")
    if counts.max() >= VOCAB_SIZE or valid.max() >= VOCAB_SIZE:
        raise ValueError("Packed token ID exceeds model vocabulary")
    manifest = {
        "source": SOURCE,
        "config": CONFIG,
        "tokenizer": "tiktoken:gpt2",
        "document_separator_id": EOT,
        "vocab_size": VOCAB_SIZE,
        "target_train_tokens": target_train_tokens,
        "elapsed_seconds": round(prior_elapsed + time.monotonic() - start, 2),
        **counters,
        "max_train_token_id": int(counts.max()),
        "max_validation_token_id": int(valid.max()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    state_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2), flush=True)
    # Let the Hugging Face/Arrow prefetch worker finish after an early stop.
    time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-train-tokens", type=int, default=16_000_000)
    parser.add_argument("--all", action="store_true", help="clean and tokenize the entire sample-10BT stream")
    parser.add_argument("--progress-interval", type=float, default=30)
    args = parser.parse_args()
    if args.target_train_tokens <= 0 or args.progress_interval <= 0:
        parser.error("target tokens and progress interval must be positive")
    prepare(args.output_dir, None if args.all else args.target_train_tokens, args.progress_interval)
