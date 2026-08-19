import torch
import torch.nn.functional as F
import tiktoken
from single_ssm import PureSSMLanguageModel  
import os

def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- NEW BULLETPROOF PATH LOGIC ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Assuming generate.py is in the 'pure' folder, go up two levels to Nano-SSM
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../")) 
    
    # Build the exact path to the checkpoint
    # Make sure 'ep10' or 'ep20' matches exactly what you have!
    CKPT_PATH = os.path.join(ROOT_DIR, "pure_ssm_ckpt", "mamba_nano_epoch_5.pt")
    # ----------------------------------
    
    DIM = 256            
    NUM_LAYERS = 4       
    
    print(f"Loading checkpoint from {CKPT_PATH}...")
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"Error: Could not find {CKPT_PATH}. Check your model folder to ensure it saved correctly.")
        return
    
    enc = tiktoken.get_encoding("gpt2")
    
    # --- FIX 1: Hardcode the padded vocab size used in training ---
    vocab_size = 50304

    # 2. BUILD THE NEW BLUEPRINT
    model = PureSSMLanguageModel(
        vocab_size=vocab_size, 
        dim=DIM, 
        num_layers=NUM_LAYERS
    ).to(device)
    
    # --- FIX 2: Load the raw weights directly ---
    model.load_state_dict(checkpoint)
    model.eval()
    
    prompt = "civil war"
    print(f"\nPrompt: {prompt}\n")
    print("--- Generating ---")
    
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    max_new_tokens = 300
    
    # --- GENERATION DIALS ---
    temperature = 0.8  # Lower = more focused, Higher = more random/creative
    top_k = 10         # How many of the top words to consider
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # 1. Apply Temperature
            next_token_logits = next_token_logits / temperature
            
            # 2. Apply Top-K filtering
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # 3. Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 4. Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append it to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Print as it generates
            word = enc.decode([next_token.item()])
            print(word, end="", flush=True)

if __name__ == "__main__":
    generate()