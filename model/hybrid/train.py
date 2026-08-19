import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
import os
import numpy as np

# 1. IMPORT YOUR NEW HYBRID ARCHITECTURE

from ssm_full import HybridLanguageModel 

torch.set_float32_matmul_precision('high')

def get_batch(data, seq_len, batch_size):
    # Create the random indexes directly on the GPU
    ix = torch.randint(len(data) - seq_len, (batch_size,), device=data.device)
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y  # Removed .to(device) because it's already there!

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Firing up training on {device.upper()} ---")

# ----------------------------------------------------------------
    # 2. HYPERPARAMETERS (The Fast Hybrid Profile)
    # ----------------------------------------------------------------
    
    BATCH_SIZE = 4       # Halved to make VRAM room for the deeper network
    SEQ_LEN = 256        
    ACCUM_STEPS = 16      # Doubled so your Effective Batch Size stays at a perfect 64
    DIM = 256            
    NUM_LAYERS = 10      # Tripled from 4 to 12. This is what perfectly triples your runtime!
    ATTN_EVERY = 5       # Now creates THREE Attention anchors throughout the network
    EPOCHS = 20

    vocab_size = 50304
    # Checkpoint to resume from (Set to None if starting fresh)
    # Example: RESUME_FILE = "model/hybridPT/wiki/hybrid_code_ckpt_ep2.pt"
    RESUME_FILE = None 
    START_EPOCH = 0
    # ----------------------------------------------------------------

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    FILE_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "train.bin"))

# ----------------------------------------------------------------
    # SUPER-FAST BINARY DATA LOADER
    # ----------------------------------------------------------------
    print(f"Loading pre-tokenized dataset from {FILE_PATH}...")
    try:
        # Read the binary file as 16-bit integers and instantly convert to PyTorch tensor
        raw_data = np.fromfile(FILE_PATH, dtype=np.uint16)
        data = torch.tensor(raw_data, dtype=torch.long).to(device)
    except FileNotFoundError:
        print(f"Error: Could not find {FILE_PATH}. Make sure your path is correct!")
        return

    print(f"Total tokens in dataset: {len(data):,}")

    # Initialize the new Hybrid Model
    print("Building the Hybrid Model (Selective SSM + Attention)...")
    model = HybridLanguageModel(
        vocab_size=vocab_size, 
        dim=DIM, 
        num_layers=NUM_LAYERS,
        attn_every=ATTN_EVERY
    ).to(device)

    model = torch.compile(model, fullgraph=True, mode="max-autotune")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    # ----------------------------------------------------------------
    # 3. RESUME ENGINE
    # ----------------------------------------------------------------
    if RESUME_FILE and os.path.exists(RESUME_FILE):
        print(f"\n[*] Waking model up from {RESUME_FILE}...")
        checkpoint = torch.load(RESUME_FILE, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1  # Start at the next epoch
        print(f"[*] Successfully loaded state! Resuming from Epoch {START_EPOCH + 1}...\n")
    # ----------------------------------------------------------------

    batches_per_epoch = len(data) // (BATCH_SIZE * SEQ_LEN)
    print(f"Starting Training! Total batches per epoch: {batches_per_epoch:,}")
    print(f"Total Epochs: {EPOCHS} | Total Training Steps: {batches_per_epoch * EPOCHS:,}\n")

    for epoch in range(START_EPOCH, EPOCHS): 
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i in range(batches_per_epoch):
            x, y = get_batch(data, SEQ_LEN, BATCH_SIZE)
            
            # Automatic Mixed Precision for speed
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            # Update weights only after accumulating enough gradients
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUM_STEPS
            
            # Print an update every 100 batches
            if i % 100 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch {epoch+1} | Batch {i:5d}/{batches_per_epoch} | Loss: {avg_loss:.4f}")
                
        # Save the checkpoint safely!
        SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "model", "hybridPT", "wiki"))
        os.makedirs(SAVE_DIR, exist_ok=True)

        ckpt_path = os.path.join(SAVE_DIR, f"hybrid_code_ckpt_ep{epoch+1}.pt")        
        # Ensure the directory exists before saving
        
        print(f"\nSaving model to {ckpt_path}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / batches_per_epoch,
        }, ckpt_path)
        print("Save complete!\n")

if __name__ == "__main__":
    train()