import os
import tiktoken
import numpy as np
from datasets import load_dataset

def download_and_prepare():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    bin_path = os.path.join(data_dir, "train.bin")
    
    # 1. Download via Hugging Face (handles all the server/extraction stuff safely)
    print("Downloading WikiText-103 from Hugging Face...")
    # This downloads and caches the dataset automatically
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    
    # 2. Tokenize and save as binary
    print("Tokenizing... (This is 100+ million words, grab a coffee)")
    enc = tiktoken.get_encoding("gpt2")
    
    # Stream the dataset line by line to keep RAM usage near zero!
    with open(bin_path, 'wb') as f_out:
        for item in dataset:
            text = item['text']
            if text.strip():  # Skip empty lines
                tokens = enc.encode(text)
                # Write straight to the hard drive
                f_out.write(np.array(tokens, dtype=np.uint16).tobytes())
                
    print(f"\nDone! Dataset fully packed into {bin_path}.")
    print("You are ready to start training!")

if __name__ == "__main__":
    download_and_prepare()