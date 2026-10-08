# Pure and hybrid SSM training

The pure model runs through `model/pure/train_ssm.py`; the hybrid model runs
through `model/hybrid/train.py`. Both use `model/ssm_scan.py` for the SSM. On
CUDA, that module selects its Triton scan if Triton is importable; otherwise
it uses the dense PyTorch fallback. `torch.compile` is independent of this
selection.

On the current Jetson setup, Triton is in `/tmp/ssm-triton-target`. Prefix
commands with `PYTHONPATH=/tmp/ssm-triton-target` and use the `ssm` Python
environment to run the Triton path. That temporary path may need to be
recreated after a restart.

Training defaults to fused AdamW and does not checkpoint block activations.
These optional settings apply to both training scripts:

```bash
PYTHONPATH=/tmp/ssm-triton-target SSM_OPTIMIZER=adafactor SSM_CHECKPOINT_BLOCKS=1 \
SSM_CKPT_DIR=/path/to/new-checkpoint-directory \
/home/nishantj/miniforge3/envs/ssm/bin/python model/pure/train_ssm.py
```

Adafactor saves optimizer memory but uses a different update rule. Use a new
checkpoint directory when switching optimizers; checkpoints record the
optimizer name and cannot resume with a different optimizer. Block activation
checkpointing recomputes each block during backward to save activation memory.
`SSM_DIM`, `SSM_LAYERS`, `SSM_BATCH_SIZE`, and `SSM_ACCUM_STEPS` can also
override the default 512-wide, eight-layer, batch-four configuration. Use a
new checkpoint directory when changing model dimensions. The larger shapes
in the benchmark are fit tests; the current training corpus is much too small
to train a multi-billion-parameter model well from scratch.

To compare configurations without changing a training script:

```bash
PYTHONPATH=/tmp/ssm-triton-target /home/nishantj/miniforge3/envs/ssm/bin/python benchmark_ssm_scan.py --model pure --backend triton --optimizer adamw --seconds 30
PYTHONPATH=/tmp/ssm-triton-target /home/nishantj/miniforge3/envs/ssm/bin/python benchmark_ssm_scan.py --model pure --backend triton --optimizer adafactor --checkpoint-blocks --seconds 30
```

## FineWeb sample-10BT

`data/prepare_fineweb.py` streams the `HuggingFaceFW/fineweb` `sample-10BT`
split. It rejects short, low English score, mostly tabular, or highly repeated
pages; removes common page boilerplate and literal unknown-token markers; and
keeps readable heading text while removing heading markup. It tokenizes with
GPT-2's byte-level tokenizer, inserts token ID 50256 between documents, and
checks that every ID fits the model's 50304-token vocabulary. A stable 1% of
documents goes to validation. This is a heuristic filter, so inspect samples
before a long run.

Prepare enough tokens for a timed pilot, then run the pure model on sequential
windows of those tokens:

```bash
/home/nishantj/miniforge3/envs/ssm/bin/python data/prepare_fineweb.py \
  --output-dir data/fineweb_10bt_clean --target-train-tokens 16000000
PYTHONPATH=/tmp/ssm-triton-target \
/home/nishantj/miniforge3/envs/ssm/bin/python benchmark_ssm_scan.py \
  --model pure --backend triton --optimizer adamw --seconds 600 \
  --data-path data/fineweb_10bt_clean/train.bin
```

Use `--all` during preparation to stream and tokenize the entire sample-10BT
split. Cleaning removes some source documents and tokens, so the resulting
binary corpus will contain fewer than the source's roughly 10 billion GPT-2
tokens. For the regular pure training script, set
`SSM_DATA_PATH=data/fineweb_10bt_clean/train.bin`.

## Optional fused output loss for the pure model

The pure model can compute its training loss with Liger's fused linear
cross-entropy, avoiding the full `[batch * sequence, vocabulary]` logits
tensor. This preserves the model's normal `forward()` logits output and all
checkpoint parameter names. It is optional because the best choice depends on
batch size. On this Jetson, it was slower at batch 4, but at batch 32 a repeated
30-second FineWeb comparison measured 33.5k versus 30.4k tokens/s and 4.0
versus 15.5 GiB of reserved CUDA memory. Short runs vary with device clocks;
check validation quality when changing the effective batch size.

Liger Kernel 0.8.4 was tested with this setup. With the isolated installation
used here, a new pure training run can select it as follows:

```bash
PYTHONPATH=/tmp/ssm-liger-target:/tmp/ssm-triton-target \
SSM_FUSED_LOSS=1 SSM_BATCH_SIZE=32 SSM_ACCUM_STEPS=1 \
SSM_DATA_PATH=data/fineweb_10bt_clean/train.bin \
SSM_CKPT_DIR=pure_ssm_fineweb_fused_ckpt \
/home/nishantj/miniforge3/envs/ssm/bin/python model/pure/train_ssm.py
```

For a benchmark without changing the regular training configuration, add
`--fused-loss` to `benchmark_ssm_scan.py`. Install `liger-kernel==0.8.4` into
the environment, or set `PYTHONPATH` to an isolated installation as above.

For the 10-minute batch-32 FineWeb benchmark, prepare at least 40 million
training tokens so sequential windows do not repeat, then run:

```bash
/home/nishantj/miniforge3/envs/ssm/bin/python data/prepare_fineweb.py \
  --output-dir data/fineweb_10bt_fused40m --target-train-tokens 40000000
PYTHONPATH=/tmp/ssm-liger-target:/tmp/ssm-triton-target \
/home/nishantj/miniforge3/envs/ssm/bin/python benchmark_ssm_scan.py \
  --model pure --backend triton --optimizer adamw --fused-loss \
  --batch-size 32 --seconds 600 \
  --data-path data/fineweb_10bt_fused40m/train.bin
```

The run on this Jetson processed 18.97 million fresh tokens in 600 seconds
(31.6k tokens/s), with 4.0 GiB peak reserved CUDA memory. The benchmark does
not include the regular trainer's periodic checkpoint writes, learning-rate
scheduler, or gradient clipping.

## Full FineWeb run

`data/prepare_fineweb.py --all` streams and cleans the entire sample-10BT
split into a uint16 training file and a document-level held-out split. The
`model/pure/train_fineweb_full.py` runner verifies that the full training file
starts with the exact pilot token stream before it resumes the batch-32 fused
checkpoint. It uses sequential windows, gradient clipping, a cosine learning
rate decay from 3e-4 to 3e-5, validation every 5,000 updates, and an atomic
checkpoint every 5,000 updates. `--passes 1` stops after one full sweep;
`--passes 4` trains four sweeps. A restart resumes from
`latest_checkpoint.pt` without replaying earlier windows.

For a new pure model on the same prepared corpus, use `--init fresh` instead
of the pilot checkpoint and specify its width and layer count. The checkpoint
records these settings and the batch shape so a restart rejects an incompatible
configuration. For example, the 343M-parameter trial used:

```bash
PYTHONPATH=.deps/liger:.deps/triton TRITON_CACHE_DIR=.cache/triton \
/home/nishantj/miniforge3/envs/ssm/bin/python model/pure/train_fineweb_full.py \
  --data-path data/fineweb_10bt_full/train.bin \
  --validation-path data/fineweb_10bt_full/validation.bin \
  --output-dir data/fineweb_10bt_full/training_balanced_343m \
  --init fresh --dim 1536 --layers 8 --passes 4 \
  --batch-size 64 --seq-len 512
```

The local `ssm-fineweb-prep.service` prepares the corpus. The
`ssm-fineweb-train.service` waits for preparation and the selected pass count in
`data/fineweb_10bt_full/run_config.json`, then runs the trainer and records
jtop samples. Check status with `systemctl --user status ssm-fineweb-prep.service`
and `systemctl --user status ssm-fineweb-train.service`. The generated data,
logs, and checkpoints live under `data/fineweb_10bt_full/`.
After the selected passes finish, `model/pure/evaluate_fineweb.py` scores the
entire held-out split and writes loss, perplexity, token accuracy, and
evaluation throughput to `data/fineweb_10bt_full/training/evaluation.json`.

The 343M configuration runs through
`data/systemd/ssm-fineweb-balanced.service`, which invokes
`data/supervise_fineweb_balanced.py`. Its logs, jtop samples, checkpoints, and
final evaluation go to `data/fineweb_10bt_full/training_balanced_343m/`.
The service restarts on failure, and the trainer resumes from its latest
checkpoint with the same data cursor and optimizer state.
