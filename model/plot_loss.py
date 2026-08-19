import matplotlib.pyplot as plt
import re

def get_epoch_losses(filename):
    epoch_losses = {}
    # This regex hunts for the words "Epoch X/... Loss: Y.YYY" in your text files
    pattern = re.compile(r"Epoch (\d+)/.*?Loss:\s*([0-9.]+)")
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    ep = int(match.group(1))
                    loss = float(match.group(2))
                    
                    if ep not in epoch_losses:
                        epoch_losses[ep] = []
                    epoch_losses[ep].append(loss)
        
        # Calculate the average loss for each epoch
        return [sum(losses)/len(losses) for ep, losses in sorted(epoch_losses.items())]
    except FileNotFoundError:
        print(f"Could not find {filename}. Make sure you saved it!")
        return []

# 1. Read the data
hybrid_losses = get_epoch_losses('hybrid_log.txt')
pure_losses = get_epoch_losses('pure_log.txt')

# 2. Build the graph
plt.figure(figsize=(10, 6))
plt.style.use('dark_background') # Makes it look like a cool terminal graph

if hybrid_losses:
    plt.plot(range(1, len(hybrid_losses)+1), hybrid_losses, 
             label='Hybrid SSM (With Attention)', color='cyan', marker='o', linewidth=2)
if pure_losses:
    plt.plot(range(1, len(pure_losses)+1), pure_losses, 
             label='Pure Multi-Head SSM', color='magenta', marker='s', linewidth=2)

# 3. Format the graph
plt.title('Model A/B Test: Hybrid vs Pure SSM', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss (Lower is Better)', fontsize=12)
plt.xticks(range(1, max(len(hybrid_losses), len(pure_losses)) + 1))
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

# 4. Show it!
plt.tight_layout()
plt.show()