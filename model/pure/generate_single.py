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
    # Default to latest checkpoint or epoch 3
    CKPT_PATH = os.path.join(ROOT_DIR, "pure_ssm_ckpt", "latest_checkpoint.pt")
    if not os.path.exists(CKPT_PATH):
        CKPT_PATH = os.path.join(ROOT_DIR, "pure_ssm_ckpt", "mamba_nano_epoch_3.pt")
    # ----------------------------------
    
    DIM = 512            
    NUM_LAYERS = 8       
    
    print(f"Loading checkpoint from {CKPT_PATH}...")
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
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
    
    # --- FIX 2: Load weights (handling both raw state_dict and checkpoint dict) ---
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[*] Loaded checkpoint: Epoch {checkpoint.get('epoch')}, Update {checkpoint.get('completed_updates')}, Loss {checkpoint.get('loss', 0):.4f}")
    else:
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
    stop_sequences = ["<|endoftext|>", "\n\n="]
    
    generated_text = ""
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Clamp to max context length (512) to avoid OOM
            cond_input = input_ids[:, -512:]
            logits = model(cond_input)
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
            
            # Stop immediately if EOS/Boundary token is generated
            if next_token.item() == enc.eot_token:
                print("\n[Reached <|endoftext|>]")
                break
                
            # Append it to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Print as it generates
            word = enc.decode([next_token.item()])
            generated_text += word
            print(word, end="", flush=True)
            
            # Stop if any text stop sequence was hit (e.g. starting a new Wikipedia article)
            if any(stop_seq in generated_text for stop_seq in stop_sequences):
                print("\n[Stopped at section/article boundary]")
                break
    print()

if __name__ == "__main__":
    generate()