import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PURE_DIR = os.path.join(SCRIPT_DIR, "model", "pure")
if PURE_DIR not in sys.path:
    sys.path.insert(0, PURE_DIR)

from single_ssm import PureSSMLanguageModel

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== BENCHMARKING PURE SSM MODEL ON {device.upper()} ===")
    
    # Paths
    ROOT_DIR = SCRIPT_DIR
    CKPT_PATH = os.path.join(ROOT_DIR, "pure_ssm_ckpt", "latest_checkpoint.pt")
    DATA_PATH = os.path.join(ROOT_DIR, "data", "train.bin")
    
    DIM = 512
    NUM_LAYERS = 8
    vocab_size = 50304
    
    # 1. Load Checkpoint
    print(f"Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    
    model = PureSSMLanguageModel(
        vocab_size=vocab_size,
        dim=DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        updates = ckpt.get('completed_updates', '?')
        loss = ckpt.get('loss', 0.0)
        print(f"[*] Loaded checkpoint successfully: Epoch {epoch}, Update {updates}, Loss {loss:.4f}")
    else:
        model.load_state_dict(ckpt)
        print("[*] Loaded raw state dict checkpoint.")
        
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    # 2. Perplexity Evaluation on Cleaned Data
    print("\n" + "="*60)
    print("1. PERPLEXITY & LOSS BENCHMARK")
    print("="*60)
    
    ppl_results = []
    if os.path.exists(DATA_PATH):
        raw_data = np.fromfile(DATA_PATH, dtype=np.uint16)
        data = torch.from_numpy(raw_data).long()
        seq_len = 512
        eval_batches = 20
        total_eval_loss = 0.0
        
        torch.manual_seed(42)
        with torch.no_grad():
            for b in range(eval_batches):
                start_idx = torch.randint(0, len(data) - seq_len - 1, (1,)).item()
                x = data[start_idx:start_idx+seq_len].unsqueeze(0).to(device)
                y = data[start_idx+1:start_idx+seq_len+1].unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    step_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                    total_eval_loss += step_loss.item()
                    
        avg_eval_loss = total_eval_loss / eval_batches
        perplexity = math.exp(avg_eval_loss)
        print(f"Validation Sequences Evaluated: {eval_batches} (Sequence Length: {seq_len})")
        print(f"Average Cross-Entropy Loss:    {avg_eval_loss:.4f}")
        print(f"Perplexity (PPL):               {perplexity:.2f}")
    else:
        print(f"Warning: {DATA_PATH} not found.")

    # 3. Prompt Generation & Latency Benchmark
    print("\n" + "="*60)
    print("2. INPUT PROMPT GENERATION & SPEED BENCHMARK")
    print("="*60)
    
    test_prompts = [
        " = The Roman Empire = \n",
        " = World War II = \n The conflict began",
        "The Industrial Revolution transformed",
        "In the solar system, Mars is",
        "Artificial intelligence is rapidly",
        "Deep beneath the ocean surface,"
    ]
    
    benchmarks = []
    temperature = 0.8
    top_k = 10
    max_tokens_to_generate = 60
    
    for prompt_idx, prompt in enumerate(test_prompts, 1):
        prompt_tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Warmup / Pre-fill timing (Time To First Token)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        with torch.no_grad():
            cond_input = input_ids[:, -512:]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(cond_input)
            next_token_logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            first_token = torch.multinomial(probs, num_samples=1)
            
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t_start) * 1000.0  # ms
        
        # Generation loop timing
        input_ids = torch.cat([input_ids, first_token], dim=1)
        generated_tokens = [first_token.item()]
        
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            for step in range(max_tokens_to_generate - 1):
                cond_input = input_ids[:, -512:]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(cond_input)
                next_token_logits = logits[:, -1, :] / temperature
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == enc.eot_token:
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
        torch.cuda.synchronize()
        gen_duration = time.perf_counter() - t_gen_start
        total_gen_tokens = len(generated_tokens)
        throughput = (total_gen_tokens - 1) / max(0.001, gen_duration) if total_gen_tokens > 1 else 0.0
        
        output_text = enc.decode(generated_tokens)
        
        benchmarks.append({
            "prompt": prompt.replace("\n", "\\n"),
            "prompt_len": len(prompt_tokens),
            "ttft_ms": ttft,
            "gen_tokens": total_gen_tokens,
            "throughput_tok_s": throughput,
            "sample_output": output_text.replace("\n", " ").strip()
        })
        
        print(f"\n--- [Prompt {prompt_idx}/{len(test_prompts)}] \"{prompt.strip()}\" ---")
        print(f"Prompt Tokens: {len(prompt_tokens)} | TTFT: {ttft:.1f} ms | Generated: {total_gen_tokens} tokens | Speed: {throughput:.1f} tok/s")
        print(f"Output: {prompt}{output_text}\n")
        
    print("\n" + "="*60)
    print("3. BENCHMARK SUMMARY TABLE")
    print("="*60)
    print(f"{'Prompt':<35} | {'Prompt Tok':<10} | {'TTFT (ms)':<10} | {'Gen Tok':<8} | {'Speed (tok/s)':<12}")
    print("-" * 85)
    for b in benchmarks:
        print(f"{b['prompt'][:33]:<35} | {b['prompt_len']:<10} | {b['ttft_ms']:<10.1f} | {b['gen_tokens']:<8} | {b['throughput_tok_s']:<12.1f}")
    
    avg_ttft = sum(b['ttft_ms'] for b in benchmarks) / len(benchmarks)
    avg_speed = sum(b['throughput_tok_s'] for b in benchmarks) / len(benchmarks)
    print("-" * 85)
    print(f"{'AVERAGE':<35} | {'-':<10} | {avg_ttft:<10.1f} | {'-':<8} | {avg_speed:<12.1f}")

if __name__ == "__main__":
    run_benchmark()
