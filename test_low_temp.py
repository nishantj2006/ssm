import os
import sys
import torch
import torch.nn.functional as F
import tiktoken

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PURE_DIR = os.path.join(SCRIPT_DIR, "model", "pure")
if PURE_DIR not in sys.path:
    sys.path.insert(0, PURE_DIR)

from single_ssm import PureSSMLanguageModel

def run_low_temp_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== LOW TEMPERATURE COMPARISON (0.5 vs 0.3 vs Greedy/0.0) ON {device.upper()} ===")
    
    ROOT_DIR = SCRIPT_DIR
    CKPT_PATH = os.path.join(ROOT_DIR, "pure_ssm_ckpt", "latest_checkpoint.pt")
    
    DIM = 512
    NUM_LAYERS = 8
    vocab_size = 50304
    
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = PureSSMLanguageModel(vocab_size=vocab_size, dim=DIM, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    
    prompts = [
        ("Valkyria Chronicles III", " = Valkyria Chronicles III = \n Senjō no Valkyria 3 : Unrecorded Chronicles is a tactical role @-@ playing video game developed by"),
        ("Ancient Egyptian Deities", " = Ancient Egyptian deities = \n Ancient Egyptian deities are the gods and goddesses worshipped in ancient Egypt . The beliefs and rituals surrounding these gods formed the core of ancient Egyptian religion , which emerged sometime in"),
        ("USS Atlanta (1861)", " = USS Atlanta ( 1861 ) = \n Atlanta was a casemate ironclad that served in the Confederate and Union navies during the American Civil War . She was converted from a Scottish @-@ built blockade runner named Fingal by"),
        ("Columbus Blue Jackets", " = 2011 – 12 Columbus Blue Jackets season = \n The 2011 – 12 Columbus Blue Jackets season was the team 's 12th season in the National Hockey League ( NHL ) . The Blue Jackets ' record of")
    ]
    
    temperatures = [0.5, 0.3, 0.0]
    max_tokens = 60
    
    for prompt_title, prompt_text in prompts:
        print("\n" + "#" * 80)
        print(f"PROMPT TOPIC: {prompt_title}")
        print(f"PROMPT TEXT:\n{prompt_text}")
        print("#" * 80)
        
        prompt_tokens = enc.encode(prompt_text, allowed_special={"<|endoftext|>"})
        
        for temp in temperatures:
            label = f"Greedy (T = 0.0)" if temp == 0.0 else f"Temperature T = {temp}"
            print(f"\n--- {label} ---")
            
            input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
            generated_words = []
            
            with torch.no_grad():
                for step in range(max_tokens):
                    cond_input = input_ids[:, -512:]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(cond_input)
                    next_token_logits = logits[:, -1, :]
                    
                    if temp == 0.0:
                        # Pure greedy argmax
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    else:
                        scaled_logits = next_token_logits / temp
                        # Top-k sampling (top 5 for tight focus)
                        v, _ = torch.topk(scaled_logits, min(5, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float('inf')
                        probs = F.softmax(scaled_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                    if next_token.item() == enc.eot_token:
                        print(" [<|endoftext|>]", end="")
                        break
                        
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    word = enc.decode([next_token.item()])
                    generated_words.append(word)
                    print(word, end="", flush=True)
            print()

if __name__ == "__main__":
    run_low_temp_experiments()
