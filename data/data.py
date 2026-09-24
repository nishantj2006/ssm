import os
import re
import tiktoken
import numpy as np
from datasets import load_dataset

def download_and_prepare(dataset_name="wikitext-2-raw-v1"):
    data_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(data_dir, "train.bin")
    
    print(f"[*] Downloading and loading Salesforce/wikitext ({dataset_name})...")
    dataset = load_dataset("Salesforce/wikitext", dataset_name, split="train")
    
    print("[*] Initializing GPT-2 BPE tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token  # 50256: <|endoftext|>
    print(f"[*] EOS/Boundary token: ID {eot_token} (<|endoftext|>)")
    
    # Regex for top-level Wikipedia article titles: " = Article Title = \n"
    # Matches single '=' on each side, distinguishing from subheadings '== Section =='
    article_header_re = re.compile(r"^ = [^=]+ = \n$")
    
    articles = []
    current_doc = []
    
    print("[*] Segmenting corpus into individual articles...")
    for item in dataset:
        text = item["text"]
        if article_header_re.match(text):
            if current_doc:
                doc_str = "".join(current_doc).strip()
                if doc_str:
                    articles.append(doc_str)
                current_doc = []
        current_doc.append(text)
        
    if current_doc:
        doc_str = "".join(current_doc).strip()
        if doc_str:
            articles.append(doc_str)
            
    print(f"[*] Successfully extracted {len(articles)} distinct documents.")
    
    print(f"[*] Tokenizing documents and appending <|endoftext|> boundaries to {bin_path}...")
    total_tokens = 0
    with open(bin_path, "wb") as f_out:
        for idx, doc in enumerate(articles):
            tokens = enc.encode(doc, allowed_special={"<|endoftext|>"})
            tokens.append(eot_token)
            total_tokens += len(tokens)
            f_out.write(np.array(tokens, dtype=np.uint16).tobytes())
            if (idx + 1) % 100 == 0 or (idx + 1) == len(articles):
                print(f"    Processed {idx + 1}/{len(articles)} articles ({total_tokens:,} tokens)...")
                
    print(f"\n[+] Done! Dataset packed into {bin_path}")
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Documents separated: {len(articles)}")
    print(f"    Boundary tokens added: {len(articles)}")
    print(f"    <unk> tokens: 0")

if __name__ == "__main__":
    download_and_prepare()