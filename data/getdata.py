import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "dataset.txt"

print(f"Downloading TinyShakespeare from {url}...")
urllib.request.urlretrieve(url, filename)

size_mb = os.path.getsize(filename) / (1024 * 1024)
print(f"Success! Saved to {filename} ({size_mb:.2f} MB)")