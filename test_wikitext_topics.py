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

def run_tests():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TESTING WIKITEXT-2 CORPUS TOPICS (HIGHER TEMPERATURE) ON {device.upper()} ===")
    
    ROOT_DIR = SCRIPT_DIR
    CKPT_PATH = os.path.join(ROOT_DIR, "pure_ssm_ckpt", "latest_checkpoint.pt")
    
    DIM = 512
    NUM_LAYERS = 8
    vocab_size = 50304
    
    print(f"Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    
    model = PureSSMLanguageModel(
        vocab_size=vocab_size,
        dim=DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[*] Loaded checkpoint: Epoch {ckpt.get('epoch')}, Update {ckpt.get('completed_updates')}, Loss {ckpt.get('loss', 0):.4f}")
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    # 4 distinct topics strictly from the WikiText-2 training corpus
    prompts = [
        ("Valkyria Chronicles III (Video Game)", " = Valkyria Chronicles III = \n Senjō no Valkyria 3 : Unrecorded Chronicles is a tactical role @-@ playing video game developed by"),
        ("Ancient Egyptian Deities (Mythology)", " = Ancient Egyptian deities = \n Ancient Egyptian deities are the gods and goddesses worshipped in ancient Egypt . The beliefs and rituals surrounding these gods formed the core of ancient Egyptian religion , which emerged sometime in"),
        ("USS Atlanta 1861 (Naval Ironclad)", " = USS Atlanta ( 1861 ) = \n Atlanta was a casemate ironclad that served in the Confederate and Union navies during the American Civil War . She was converted from a Scottish @-@ built blockade runner named Fingal by"),
        ("Columbus Blue Jackets (NHL Hockey)", " = 2011 – 12 Columbus Blue Jackets season = \n The 2011 – 12 Columbus Blue Jackets season was the team 's 12th season in the National Hockey League ( NHL ) . The Blue Jackets ' record of")
    ]
    
    # Higher temperature (1.0 vs previous 0.8) and broader sampling pool
    temperature = 1.05
    top_k = 25
    max_new_tokens = 90
    stop_sequences = ["<|endoftext|>", "\n\n\n"]
    
    print(f"\n[Settings] Temperature: {temperature} (Higher/Creative) | Top-K: {top_k} | Max New Tokens: {max_new_tokens}\n")
    
    results = []
    
    for idx, (title, prompt) in enumerate(prompts, 1):
        print(f"\n{'='*75}")
        print(f"Input {idx}/4: [{title}]")
        print(f"Prompt: {prompt}")
        print(f"{'-'*75}")
        print("Generated Output:\n", flush=True)
        
        input_ids = torch.tensor(enc.encode(prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long).unsqueeze(0).to(device)
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cond_input = input_ids[:, -512:]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(cond_input)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-K filtering
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == enc.eot_token:
                    print("\n[<|endoftext|>]", flush=True)
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=1)
                word = enc.decode([next_token.item()])
                generated_tokens.append(word)
                print(word, end="", flush=True)
                
                # Check stop sequences
                full_gen = "".join(generated_tokens)
                if any(stop in full_gen for stop in stop_sequences):
                    break
                    
        print("\n", flush=True)
        results.append((title, prompt, "".join(generated_tokens)))
        
    print(f"\n{'='*75}")
    print("ALL 4 TESTS COMPLETED")
    print(f"{'='*75}")

if __name__ == "__main__":
    run_tests()
