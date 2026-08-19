import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Import the new Pure SSM instead of the Hybrid
from model.heads.pure_ssm import PureSSMLanguageModel

class TextDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = enc.encode(raw_text)
        self.vocab_size = enc.n_vocab
        self.num_samples = len(self.tokens) - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        return chunk_tensor[:-1], chunk_tensor[1:]

def train():
    FILE_PATH = "dataset.txt" 
    BATCH_SIZE = 8       
    ACCUM_STEPS = 8      
    SEQ_LEN = 128        
    DIM = 16             
    NUM_LAYERS = 4       
    NUM_HEADS = 4        # The multi-head SSM setting
    EPOCHS = 10          
    MAX_LR = 5e-4
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TextDataset(FILE_PATH, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

    print("Initializing Pure Multi-Head SSM...")
    model = PureSSMLanguageModel(
        vocab_size=dataset.vocab_size, 
        dim=DIM, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS
    ).to(device)
    
    try:
        model = torch.compile(model)
        print("Model compiled with Triton.")
    except Exception as e:
        print(f"Skipping torch.compile: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = y.view(B*T)
                
                loss = criterion(logits, targets)
                loss = loss / ACCUM_STEPS 
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            true_loss = loss.item() * ACCUM_STEPS
            total_loss += true_loss
            
            if batch_idx % 50 == 0:
                print(f"Pure SSM | Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(loader)} | Loss: {true_loss:.4f}")

        avg_loss = total_loss / len(loader)
        
        # Save specifically as pure_ckpt so it doesn't ruin your hybrid model!
        checkpoint_path = f"pure_ckpt_ep{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'vocab_size': dataset.vocab_size,
            'dim': DIM,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}\n")

if __name__ == "__main__":
    train()