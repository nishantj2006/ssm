import torch
import torch.nn.functional as F
import tiktoken
import os

# 1. IMPORT YOUR EXACT HYBRID BLUEPRINT
# Make sure this matches the name of your file and the Hybrid class!
# (e.g., from hybrid_model import HybridLanguageModel)
from ssm_full import HybridLanguageModel # <-- Change 'train' to whatever file holds your model class

def generate_code(prompt="def bubble_sort(arr):", max_new_tokens=150, temperature=0.8):
    device ="cuda"
    print(f"--- Booting up Hybrid Generator on {device.upper()} ---")

    # 2. EXACT HYBRID CONSTANTS FROM YOUR TRAINING LOG
    DIM = 256
    NUM_LAYERS = 10
    SEQ_LEN = 256 
    ATTN_EVERY = 5 # Only needed if your model __init__ asks for it!
    
    # Update this to point to the folder from your terminal output!
    # I set it to ep20, but change it if you stopped it earlier.
    CKPT_PATH = r"model\hybridPT\wiki\hybrid_code_ckpt_ep9.pt" 

    if not os.path.exists(CKPT_PATH):
        print(f"Error: Could not find checkpoint at {CKPT_PATH}.")
        print("Double check the epoch number in your model folder!")
        return

    # 3. LOAD THE TOKENIZER
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = 50304

    # 4. INITIALIZE THE HYBRID ARCHITECTURE
    print("Initializing the Hybrid architecture...")
    model = HybridLanguageModel(
        vocab_size=vocab_size,
        dim=DIM,
        num_layers=NUM_LAYERS,
        attn_every=ATTN_EVERY 
    ).to(device)

    # 5. INJECT THE TRAINED WEIGHTS INTO THE MODEL
    print(f"Loading brain from {CKPT_PATH}...")
    # 1. Load the checkpoint
    checkpoint = torch.load('model/hybridPT/wiki/hybrid_code_ckpt_ep3.pt', map_location='cuda')
    state_dict = checkpoint['model_state_dict']
    
    # 2. Strip the "_orig_mod." prefix from the saved keys
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # 3. Load the cleaned dictionary into the model
    model.load_state_dict(state_dict)
    
    # Turn off training mode (saves memory and stops dropout)
    model.eval()

    # 6. PREPARE THE PROMPT
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    print("\n--- Generating Code ---\n")
    print(prompt, end="")

    # 7. THE AUTOREGRESSIVE GENERATION LOOP
    with torch.no_grad(): 
        for _ in range(max_new_tokens):
            
            # Crop context to the Hybrid's 128 SEQ_LEN limit
            cond_input = input_ids[:, -SEQ_LEN:]
            
            # Get predictions
            logits = model(cond_input)
            next_token_logits = logits[:, -1, :] 
            
            # Apply Temperature
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample and append
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Decode and print
            print(enc.decode([next_token.item()]), end="", flush=True)

    print("\n\n--- Generation Complete ---")

if __name__ == "__main__":
    generate_code(prompt="civil war", max_new_tokens=1000, temperature=0.7)