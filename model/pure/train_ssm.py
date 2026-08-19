import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.amp import autocast

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
    # HYPERPARAMETERS (The "Goldilocks" Balanced Profile)
    # ----------------------------------------------------------------
    BATCH_SIZE = 8       # A strong batch size to keep the GPU fully fed
    ACCUM_STEPS = 8      # 8 * 8 = 64 (Perfect Effective Batch Size)
    SEQ_LEN = 256        # Bumped back up so it has decent short-term memory
    DIM = 256            # The mathematical sweet spot
    NUM_LAYERS = 4       # Enough depth to learn grammar, not just spelling
    EPOCHS = 20          
    vocab_size = 50304
    
    # 1. Load the Dataset
    print(f"Loading data from {DATA_PATH}...")
    raw_data = np.fromfile(DATA_PATH, dtype=np.uint16)
    data = torch.from_numpy(raw_data).long()
    

    model = PureSSMLanguageModel(vocab_size, DIM, NUM_LAYERS).to(device)

    # ---------------------------------------------------------
    # --- THE MAGIC SPEED BOOST ---
    # ---------------------------------------------------------
    if device == "cuda":
        print("Compiling model... (The first batch will take a minute or two to start!)")
        # You can also pass mode="max-autotune" if you want to wait longer for even faster code
        model = torch.compile(model, mode="max-autotune", fullgraph=True) 
    
    # 3. Calculate Workload
    tokens_per_epoch = len(data)
    batches_per_epoch = tokens_per_epoch // (BATCH_SIZE * SEQ_LEN * ACCUM_STEPS)
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total Batches per Epoch: {batches_per_epoch}\n")

# ---------------------------------------------------------
    # 4. SETUP OPTIMIZER & SCHEDULER (The Custom SSM Tune)
    # ---------------------------------------------------------
    
    # Standard weight decay filter:
    # 2D matrices (Linear layers) get decayed. 
    # 1D tensors (log_A, LayerNorms, Biases) are safely ignored!
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    # Standard reliable learning rate for custom from-scratch architectures
    max_learning_rate = 6e-4 

    optimizer = optim.AdamW(optim_groups, lr=max_learning_rate, betas=(0.9, 0.95), fused=True)
    
    total_update_steps = (batches_per_epoch // ACCUM_STEPS) * EPOCHS
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_learning_rate,        
        total_steps=total_update_steps,
        pct_start=0.10,                  # Keep the 10% warmup just to be safe with the exp() math
        div_factor=10.0,                 
        final_div_factor=10.0
    )
    # ---------------------------------------------------------

    # 5. THE TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Start with a clean slate
        optimizer.zero_grad(set_to_none=True)
        
        for i in range(batches_per_epoch):
            # Grab data
            x, y = get_batch(data, SEQ_LEN, BATCH_SIZE, device)
            
            # --- BFLOAT16 Forward Pass ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / ACCUM_STEPS
            
            # --- Backward Pass (NO SCALER NEEDED FOR BFLOAT16) ---
            loss.backward()
            
            # --- Gradient Accumulation Update Step ---
            if (i + 1) % ACCUM_STEPS == 0:
                # 1. Clip Gradients to catch violent math spikes (The Seatbelt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 2. Step Optimizer
                optimizer.step()
                
                # 3. Step Scheduler
                scheduler.step()
                
                # 4. Empty the bucket for the next round
                optimizer.zero_grad(set_to_none=True)
            
            # Track the loss for printing (scale it back up for human reading)
            total_loss += loss.item() * ACCUM_STEPS
            
            # Print an update every 20 true update cycles
            if (i + 1) % (20 * ACCUM_STEPS) == 0:
                avg_loss = total_loss / (i + 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i+1:5d}/{batches_per_epoch} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # --- THE AUTO-SAVER (End of Epoch) ---
        print(f"\n--- Epoch {epoch+1} Complete ---")
        os.makedirs("pure_ssm_ckpt", exist_ok=True) 
        checkpoint_path = f"pure_ssm_ckpt/mamba_nano_epoch_{epoch+1}.pt"
        
        # We save ONLY the weights (state_dict) to save storage space
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[*] Saved checkpoint to {checkpoint_path}\n")

if __name__ == "__main__":
    train()