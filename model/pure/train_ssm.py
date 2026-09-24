import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.amp import autocast

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import your optimized Mamba model
from single_ssm import PureSSMLanguageModel
# Tell PyTorch to use faster matrix multiplication
torch.set_float32_matmul_precision('high')

def get_batch(data, seq_len, batch_size, device):
    # Random sampling with replacement (This is how it learns the whole file!)
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Firing up OPTIMIZED PURE SSM training on {device.upper()} ---")

    # ----------------------------------------------------------------
    # BULLETPROOF PATHS (Works on Windows & Linux automatically)
    # ----------------------------------------------------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    DATA_PATH = os.path.join(ROOT_DIR, "data", "train.bin")
    
    # ----------------------------------------------------------------
    # HYPERPARAMETERS (Max Memory Profile: ~44-48 GB VRAM Utilization)
    # ----------------------------------------------------------------
    # This configuration maximizes the context window to 512 tokens (2x original)
    # while utilizing ~44 GB of your 64 GB unified memory without triggering OS swap/OOM.
    # ----------------------------------------------------------------
    BATCH_SIZE = 4       # Micro-batch size (4 * 512 tokens = 2,048 tokens per forward pass)
    ACCUM_STEPS = 4      # 4 * 4 = 16 Effective Batch Size (frequent updates & fast tracking)
    SEQ_LEN = 512        # 2x longer context window (512 tokens)
    DIM = 512            # Model dimension (72.5M total parameters)
    NUM_LAYERS = 8       # 8 layers for deep syntax and long-range semantic representation
    EPOCHS = 20          
    vocab_size = 50304
    
    # 1. Load the Dataset
    print(f"Loading data from {DATA_PATH}...")
    raw_data = np.fromfile(DATA_PATH, dtype=np.uint16)
    data = torch.from_numpy(raw_data).long()
    

    model = PureSSMLanguageModel(vocab_size, DIM, NUM_LAYERS).to(device)

    # ---------------------------------------------------------
    # CHECKPOINT RESUME (Build off previous training runs)
    # ---------------------------------------------------------
    start_epoch = 0
    start_completed_updates = 0
    CKPT_DIR = "pure_ssm_ckpt"
    latest_ckpt_path = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
    ckpt = None
    if os.path.exists(latest_ckpt_path):
        print(f"[*] Found checkpoint: {latest_ckpt_path}! Loading weights...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = max(0, ckpt.get('epoch', 1) - 1)
            start_completed_updates = ckpt.get('completed_updates', 0)
        else:
            model.load_state_dict(ckpt)
        print(f"[*] Resumed weights! Starting at Epoch {start_epoch + 1} (update {start_completed_updates})")

    # ---------------------------------------------------------
    # --- COMPILATION (Safe Triton / TorchInductor Fallback) ---
    # ---------------------------------------------------------
    # On Jetson Orin (aarch64), Triton is typically not available,
    # so eager PyTorch CUDA with TF32 cuBLAS runs directly and reliably.
    COMPILE = False
    if COMPILE and device == "cuda":
        try:
            import triton
            print("Attempting torch.compile...")
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"Skipping torch.compile ({e}); running eager PyTorch CUDA.")
    else:
        print("Running native PyTorch CUDA execution (optimized TF32 cuBLAS).")
    
    # 3. Calculate Workload
    tokens_per_epoch = len(data)
    micro_batches_per_epoch = tokens_per_epoch // (BATCH_SIZE * SEQ_LEN)
    updates_per_epoch = max(1, micro_batches_per_epoch // ACCUM_STEPS)
    total_update_steps = updates_per_epoch * EPOCHS
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tokens in Dataset:      {tokens_per_epoch:,}")
    print(f"Micro-batches per Epoch: {micro_batches_per_epoch}")
    print(f"Optimizer Steps / Epoch: {updates_per_epoch}")
    print(f"Total Optimizer Steps:   {total_update_steps}\n")

    # ---------------------------------------------------------
    # 4. SETUP OPTIMIZER & SCHEDULER (The Custom SSM Tune)
    # ---------------------------------------------------------
    
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    max_learning_rate = 6e-4 

    optimizer = optim.AdamW(optim_groups, lr=max_learning_rate, betas=(0.9, 0.95), fused=True)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_learning_rate,        
        total_steps=total_update_steps,
        pct_start=0.10,                  # 10% warmup
        div_factor=10.0,                 
        final_div_factor=10.0
    )

    # Restore optimizer state and fast-forward scheduler if resuming
    if ckpt and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("[*] Restored AdamW optimizer states!")
        except Exception as e:
            print(f"[*] Starting fresh optimizer states ({e})")
            
    if start_completed_updates > 0:
        print(f"[*] Fast-forwarding LR scheduler to update {start_completed_updates}...")
        for _ in range(start_completed_updates):
            scheduler.step()
        print(f"[*] Resumed Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
    # ---------------------------------------------------------

    # 5. THE TRAINING LOOP
    import time
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
            step_start = time.time()
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
            
            # --- Gradient Accumulation Update Step ---
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
                tokens_processed = (i + 1) * BATCH_SIZE * SEQ_LEN
                tok_per_sec = tokens_processed / max(1.0, elapsed)
                
                print(f"[Update {completed_updates:4d}/{total_update_steps}] "
                      f"Epoch {epoch+1:2d} ({i+1:4d}/{micro_batches_per_epoch} mbatches) | "
                      f"Loss: {step_loss:.4f} | LR: {current_lr:.6f} | "
                      f"Elapsed: {elapsed_min:.2f}m | Speed: {tok_per_sec:.1f} tok/s", flush=True)
                
                # --- Periodic Checkpoint (Every 25 updates ~15 minutes) ---
                if completed_updates % 25 == 0:
                    os.makedirs("pure_ssm_ckpt", exist_ok=True)
                    ckpt_file = "pure_ssm_ckpt/latest_checkpoint.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'completed_updates': completed_updates,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': step_loss,
                    }, ckpt_file)
                    print(f"[*] Periodic checkpoint saved to {ckpt_file}", flush=True)
        
        # --- THE AUTO-SAVER (End of Epoch) ---
        print(f"\n--- Epoch {epoch+1} Complete ---")
        os.makedirs("pure_ssm_ckpt", exist_ok=True) 
        checkpoint_path = f"pure_ssm_ckpt/mamba_nano_epoch_{epoch+1}.pt"
        
        torch.save({
            'epoch': epoch + 1,
            'completed_updates': completed_updates,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / max(1, micro_batches_per_epoch),
        }, checkpoint_path)
        print(f"[*] Saved full epoch checkpoint to {checkpoint_path}\n")

if __name__ == "__main__":
    train()