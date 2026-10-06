import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

from ssm_full import HybridLanguageModel

torch.set_float32_matmul_precision('high')

def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Firing up OPTIMIZED HYBRID SSM+ATTENTION training on {device.upper()} ---")

    DATA_PATH = os.path.join(ROOT_DIR, "data", "train.bin")
    CKPT_DIR = os.path.join(ROOT_DIR, "hybrid_ssm_ckpt")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # HYPERPARAMETERS (Max Memory Profile: ~48-50 GB VRAM Utilization)
    # ----------------------------------------------------------------
    # Matched 1:1 with Pure SSM for head-to-head architectural A/B comparison
    # ----------------------------------------------------------------
    BATCH_SIZE = 4       # Micro-batch size (4 * 512 tokens = 2,048 tokens per forward pass)
    ACCUM_STEPS = 4      # Effective Batch Size = 16 (frequent updates & fast tracking)
    SEQ_LEN = 512        # 512 tokens context window
    DIM = 512            # Model dimension (73.6M total parameters)
    NUM_LAYERS = 8       # 8 layers total (6 Selective SSM layers + 2 RoPE Attention anchors)
    ATTN_LAYERS = (2, 5) # Inward anchors: Layers 3 & 6 (0-indexed 2 and 5), keeping Layers 7 & 8 as SSM
    EPOCHS = 8           # Extended to 8 full epochs (Epochs 5-8 continuation)
    vocab_size = 50304

    # 1. Load Dataset
    print(f"Loading data from {DATA_PATH}...")
    raw_data = np.fromfile(DATA_PATH, dtype=np.uint16)
    data = torch.from_numpy(raw_data).long()

    # 2. Workload Calculations
    tokens_per_epoch = len(data)
    micro_batches_per_epoch = tokens_per_epoch // (BATCH_SIZE * SEQ_LEN)
    updates_per_epoch = max(1, micro_batches_per_epoch // ACCUM_STEPS)
    total_update_steps = updates_per_epoch * EPOCHS

    print(f"Building Hybrid Language Model (Selective SSM + RoPE Flash Attention)...")
    model = HybridLanguageModel(
        vocab_size=vocab_size,
        dim=DIM,
        num_layers=NUM_LAYERS,
        attn_layers=ATTN_LAYERS
    ).to(device)

    # 3. Checkpoint Resume Engine
    start_completed_updates = 0
    latest_ckpt_path = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
    ckpt = None
    if os.path.exists(latest_ckpt_path):
        print(f"[*] Found checkpoint: {latest_ckpt_path}! Loading weights...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            start_completed_updates = ckpt.get('completed_updates', 0)
        else:
            model.load_state_dict(ckpt)

    start_epoch = start_completed_updates // updates_per_epoch
    if ckpt:
        print(f"[*] Resumed weights! Starting at Epoch {start_epoch + 1} (update {start_completed_updates}/{total_update_steps})")

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tokens in Dataset:      {tokens_per_epoch:,}")
    print(f"Micro-batches per Epoch: {micro_batches_per_epoch}")
    print(f"Optimizer Steps / Epoch: {updates_per_epoch}")
    print(f"Total Optimizer Steps:   {total_update_steps}\n")

    # 4. Setup Optimizer & Learning Rate Scheduler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    max_learning_rate = 3e-4
    if ckpt and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
        pgs = ckpt['optimizer_state_dict'].get('param_groups', [])
        if pgs and 'max_lr' in pgs[0]:
            max_learning_rate = pgs[0]['max_lr']
            print(f"[*] Loaded checkpoint max learning rate: {max_learning_rate}")

    optimizer = optim.AdamW(optim_groups, lr=max_learning_rate, betas=(0.9, 0.95), fused=True)

    INITIAL_BUDGET = 1164
    extending_run = start_completed_updates >= INITIAL_BUDGET

    if extending_run:
        # Phase 2 / Continuation: Smooth cosine decay from 1.5e-4 down to 1e-5
        if ckpt and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[*] Restored AdamW optimizer states!")
            except Exception as e:
                print(f"[*] Starting fresh optimizer states ({e})")

        continuation_lr = 1.5e-4
        for pg in optimizer.param_groups:
            pg['lr'] = continuation_lr

        ext_total_steps = max(1, total_update_steps - INITIAL_BUDGET)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ext_total_steps,
            eta_min=1e-5
        )

        if start_completed_updates > INITIAL_BUDGET:
            steps_into_ext = start_completed_updates - INITIAL_BUDGET
            if ckpt and isinstance(ckpt, dict) and 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print("[*] Restored extension scheduler state directly!")
                except Exception:
                    for _ in range(steps_into_ext):
                        scheduler.step()
            else:
                for _ in range(steps_into_ext):
                    scheduler.step()

        print(f"[*] Continuation Phase: CosineAnnealingLR (1.5e-4 -> 1e-5) for {total_update_steps - start_completed_updates} remaining steps")
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_learning_rate,
            total_steps=INITIAL_BUDGET,
            pct_start=0.08,
            div_factor=10.0,
            final_div_factor=10.0
        )

        if ckpt and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("[*] Restored AdamW optimizer states!")
            except Exception as e:
                print(f"[*] Starting fresh optimizer states ({e})")

        if ckpt and isinstance(ckpt, dict) and 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print("[*] Restored OneCycleLR scheduler state directly!")
            except Exception as e:
                print(f"[*] Fast-forwarding LR scheduler instead ({e})...")
                if start_completed_updates > 0:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for _ in range(start_completed_updates):
                            scheduler.step()
        elif start_completed_updates > 0:
            print(f"[*] Fast-forwarding LR scheduler to update {start_completed_updates}...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_completed_updates):
                    scheduler.step()

    print(f"[*] Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")

    # 5. Training Loop
    start_time = time.time()
    accum_loss = 0.0
    completed_updates = start_completed_updates

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        start_mbatch = 0
        if epoch == start_epoch and start_completed_updates > 0:
            start_mbatch = (start_completed_updates % updates_per_epoch) * ACCUM_STEPS
            if start_mbatch > 0:
                print(f"[*] Resuming Epoch {epoch+1} from micro-batch {start_mbatch}/{micro_batches_per_epoch}")

        for i in range(start_mbatch, micro_batches_per_epoch):
            x, y = get_batch(data, SEQ_LEN, BATCH_SIZE, device)

            # --- BFLOAT16 Forward Pass ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / ACCUM_STEPS

            # --- Backward Pass ---
            loss.backward()
            accum_loss += loss.item() * ACCUM_STEPS
            epoch_loss += loss.item() * ACCUM_STEPS

            # --- Gradient Accumulation Update ---
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1

                step_loss = accum_loss / ACCUM_STEPS
                accum_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                elapsed_min = elapsed / 60.0
                tokens_processed = (i + 1 - start_mbatch) * BATCH_SIZE * SEQ_LEN
                tok_per_sec = tokens_processed / max(1.0, elapsed)

                print(f"[Update {completed_updates:4d}/{total_update_steps}] "
                      f"Epoch {epoch+1:2d} ({i+1:4d}/{micro_batches_per_epoch} mbatches) | "
                      f"Loss: {step_loss:.4f} | LR: {current_lr:.6f} | "
                      f"Elapsed: {elapsed_min:.2f}m | Speed: {tok_per_sec:.1f} tok/s", flush=True)

                # Periodic Checkpoint (every 25 updates)
                if completed_updates % 25 == 0:
                    ckpt_file = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'completed_updates': completed_updates,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': step_loss,
                    }, ckpt_file)
                    print(f"[*] Periodic checkpoint saved to {ckpt_file}", flush=True)

        # End of Epoch Auto-Save
        print(f"\n--- Epoch {epoch+1} Complete ---")
        checkpoint_path = os.path.join(CKPT_DIR, f"hybrid_nano_epoch_{epoch+1}.pt")
        latest_file = os.path.join(CKPT_DIR, "latest_checkpoint.pt")

        epoch_ckpt = {
            'epoch': epoch + 1,
            'completed_updates': completed_updates,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss / max(1, micro_batches_per_epoch),
        }
        torch.save(epoch_ckpt, checkpoint_path)
        torch.save(epoch_ckpt, latest_file)
        print(f"[*] Saved full epoch checkpoint to {checkpoint_path} and {latest_file}\n", flush=True)

if __name__ == "__main__":
    train()