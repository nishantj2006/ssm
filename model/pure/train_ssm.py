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
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Firing up OPTIMIZED PURE SSM training on {device.upper()} ---")

    # ----------------------------------------------------------------
    # BULLETPROOF PATHS (Works on Windows & Linux automatically)
    # ----------------------------------------------------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    DATA_PATH = os.environ.get("SSM_DATA_PATH", os.path.join(ROOT_DIR, "data", "train.bin"))
    
    # ----------------------------------------------------------------
    # HYPERPARAMETERS
    # ----------------------------------------------------------------
    # Environment overrides make it possible to benchmark larger models
    # without changing the default checkpoint-compatible configuration.
    # ----------------------------------------------------------------
    BATCH_SIZE = int(os.environ.get("SSM_BATCH_SIZE", "4"))
    ACCUM_STEPS = int(os.environ.get("SSM_ACCUM_STEPS", "4"))
    SEQ_LEN = 512        # 2x longer context window (512 tokens)
    DIM = int(os.environ.get("SSM_DIM", "512"))
    NUM_LAYERS = int(os.environ.get("SSM_LAYERS", "8"))
    EPOCHS = 4           # 4 full epochs (~11.5 hours, ideal overnight run)
    vocab_size = 50304
    
    # 1. Load the Dataset
    print(f"Loading data from {DATA_PATH}...")
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
    if len(data) <= SEQ_LEN + 1:
        raise ValueError("Training token file is shorter than one sequence")
    if int(data.max()) >= vocab_size:
        raise ValueError("Training token file contains IDs outside the model vocabulary")
    
    # 2. Calculate Workload
    tokens_per_epoch = len(data)
    micro_batches_per_epoch = tokens_per_epoch // (BATCH_SIZE * SEQ_LEN)
    updates_per_epoch = max(1, micro_batches_per_epoch // ACCUM_STEPS)
    total_update_steps = updates_per_epoch * EPOCHS

    model = PureSSMLanguageModel(vocab_size, DIM, NUM_LAYERS).to(device)
    model.gradient_checkpointing = os.environ.get("SSM_CHECKPOINT_BLOCKS", "0") == "1"
    use_fused_loss = os.environ.get("SSM_FUSED_LOSS", "0") == "1"
    fused_loss = None
    if use_fused_loss:
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        except ImportError as exc:
            raise RuntimeError("SSM_FUSED_LOSS=1 requires liger-kernel") from exc
        fused_loss = LigerFusedLinearCrossEntropyLoss()
        print("Liger fused linear cross-entropy enabled")
    if model.gradient_checkpointing:
        print("Block activation checkpointing enabled")

    # ---------------------------------------------------------
    # CHECKPOINT RESUME (Build off previous training runs)
    # ---------------------------------------------------------
    start_completed_updates = 0
    CKPT_DIR = os.environ.get("SSM_CKPT_DIR", os.path.join(ROOT_DIR, "pure_ssm_ckpt"))
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

    # ---------------------------------------------------------
    # --- COMPILATION (Safe Triton / TorchInductor Fallback) ---
    # ---------------------------------------------------------
    # This controls the surrounding PyTorch model only. The SSM scan can use
    # Triton independently whenever Triton is importable on CUDA.
    COMPILE = False
    if COMPILE and device == "cuda":
        try:
            import triton
            print("Attempting torch.compile...")
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"Skipping torch.compile ({e}); running eager PyTorch CUDA.")
    else:
        print("Running without torch.compile; SSM scan selects Triton when available.")
    
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

    max_learning_rate = 3e-4
    if ckpt and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
        pgs = ckpt['optimizer_state_dict'].get('param_groups', [])
        if pgs and 'max_lr' in pgs[0]:
            max_learning_rate = pgs[0]['max_lr']
            print(f"[*] Loaded checkpoint max learning rate: {max_learning_rate}")

    optimizer_name = os.environ.get("SSM_OPTIMIZER", "adamw").lower()
    if optimizer_name not in ("adamw", "adafactor"):
        raise ValueError("SSM_OPTIMIZER must be 'adamw' or 'adafactor'")
    if ckpt and isinstance(ckpt, dict) and ckpt.get("optimizer_name", "adamw") != optimizer_name:
        raise ValueError("Checkpoint optimizer differs from SSM_OPTIMIZER; use a new SSM_CKPT_DIR")
    if optimizer_name == "adafactor":
        optimizer = optim.Adafactor(optim_groups, lr=max_learning_rate)
    else:
        optimizer = optim.AdamW(optim_groups, lr=max_learning_rate, betas=(0.9, 0.95), fused=True)
    print(f"Optimizer: {optimizer_name}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_learning_rate,        
        total_steps=total_update_steps,
        pct_start=0.08,                  # 8% warmup (~93 steps)
        div_factor=10.0,                 
        final_div_factor=10.0,
        cycle_momentum=optimizer_name == "adamw",
    )

    # Restore optimizer state and scheduler if resuming
    if ckpt and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"[*] Restored {optimizer_name} optimizer states!")
        except Exception as e:
            print(f"[*] Starting fresh optimizer states ({e})")
            
    if ckpt and isinstance(ckpt, dict) and 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("[*] Restored LR scheduler state directly!")
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
                if fused_loss is not None:
                    hidden = model.forward_hidden(x)
                    loss = fused_loss(model.classifier.weight, hidden.reshape(-1, DIM), y.reshape(-1))
                else:
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
                tokens_processed = (i + 1 - start_mbatch) * BATCH_SIZE * SEQ_LEN
                tok_per_sec = tokens_processed / max(1.0, elapsed)
                
                print(f"[Update {completed_updates:4d}/{total_update_steps}] "
                      f"Epoch {epoch+1:2d} ({i+1:4d}/{micro_batches_per_epoch} mbatches) | "
                      f"Loss: {step_loss:.4f} | LR: {current_lr:.6f} | "
                      f"Elapsed: {elapsed_min:.2f}m | Speed: {tok_per_sec:.1f} tok/s", flush=True)
                
                # --- Periodic Checkpoint (Every 25 updates) ---
                if completed_updates % 25 == 0:
                    os.makedirs(CKPT_DIR, exist_ok=True)
                    ckpt_file = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'completed_updates': completed_updates,
                        'model_state_dict': model.state_dict(),
                        'optimizer_name': optimizer_name,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': step_loss,
                    }, ckpt_file)
                    print(f"[*] Periodic checkpoint saved to {ckpt_file}", flush=True)
        
        # --- THE AUTO-SAVER (End of Epoch) ---
        print(f"\n--- Epoch {epoch+1} Complete ---")
        os.makedirs(CKPT_DIR, exist_ok=True) 
        checkpoint_path = os.path.join(CKPT_DIR, f"mamba_nano_epoch_{epoch+1}.pt")
        latest_file = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
        
        epoch_ckpt = {
            'epoch': epoch + 1,
            'completed_updates': completed_updates,
            'model_state_dict': model.state_dict(),
            'optimizer_name': optimizer_name,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss / max(1, micro_batches_per_epoch),
        }
        torch.save(epoch_ckpt, checkpoint_path)
        torch.save(epoch_ckpt, latest_file)
        print(f"[*] Saved full epoch checkpoint to {checkpoint_path} and {latest_file}\n", flush=True)

if __name__ == "__main__":
    train()
