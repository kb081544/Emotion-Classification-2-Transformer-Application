import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import glob
import os
import matplotlib.pyplot as plt

eval_interval = 2000
save_interval = 20000
eval_iters = 200
batch_size = 64
max_iters = 1000000
block_size = 250  # 10s with 25Hz
learning_rate = 3e-04
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_ppg_data(data_dir):
    all_data = []
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    for file_path in files:
        try:
            ppg_values = []
            with open(file_path, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    try:
                        timestamp, ppg = line.strip().split('\t')
                        ppg_values.append(float(ppg))
                    except ValueError:
                        continue

            if ppg_values:
                ppg_values = np.array(ppg_values)
                min_val = np.min(ppg_values)
                max_val = np.max(ppg_values)
                print(f"File: {file_path}")
                print(f"Min value: {min_val:.2f}, Max value: {max_val:.2f}")
                if max_val > min_val:
                    normalized_data = ((ppg_values - min_val) / (max_val - min_val) * 100).round().astype(int)
                    print(f"Normalized data range: {normalized_data.min()} to {normalized_data.max()}")
                    all_data.append(normalized_data)
                    print(f"Loaded and processed {file_path}: {len(normalized_data)} samples")
                else:
                    print(f"Skipped {file_path}: constant values")

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if len(all_data) > 0:
        combined_data = np.concatenate(all_data)
        print(f"\nTotal samples loaded: {len(combined_data)}")
        print(f"Combined data range: {combined_data.min()} to {combined_data.max()}")
        return combined_data
    else:
        raise ValueError("No data could be loaded from the specified directory")
def prepare_data(data_dir):
    data_ppg = load_ppg_data(data_dir)

    perm = np.random.permutation(len(data_ppg))
    data_ppg_rand = data_ppg[perm]

    data = torch.tensor(data_ppg_rand, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    return train_data, test_data


def get_batch(split, train_data, test_data, batch_idx=0):
    data = train_data if split == 'train' else test_data
    n_sequences = len(data) - block_size
    if n_sequences <= 0:
        raise ValueError(f"Data length ({len(data)}) must be greater than block_size ({block_size})")

    base_idx = (batch_idx * batch_size) % n_sequences
    start_indices = []
    for i in range(batch_size):
        idx = (base_idx + i) % n_sequences
        start_indices.append(idx)

    x = torch.stack([data[i:i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_indices])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class HeartGPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


@torch.no_grad()
def estimate_loss_and_accuracy(model, train_data, test_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, test_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

            logits = logits.view(-1, logits.size(-1))
            Y = Y.view(-1)

            predictions = logits.argmax(dim=-1)
            correct_predictions = (predictions == Y).float()
            accuracy = correct_predictions.mean().item()
            accuracies[k] = accuracy

        out[split] = {
            'loss': losses.mean().item(),
            'accuracy': accuracies.mean().item() * 100  # Convert to percentage
        }
    model.train()
    return out


def train_model(data_dir, model_save_dir):
    train_data, test_data = prepare_data(data_dir)
    print(f"Train data length: {len(train_data)}")

    vocab_size = 101
    model = HeartGPTModel(vocab_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_parameters} trainable parameters")

    n_sequences = len(train_data) - block_size
    total_batches = max(1, n_sequences // batch_size)
    print(f"Number of sequences: {n_sequences}")
    print(f"Total number of batches per epoch: {total_batches}")

    best_val_acc = 0.0
    global_batch_idx = 0

    for iter in range(max_iters):
        current_batch_idx = global_batch_idx

        if iter % eval_interval == 0:
            metrics = estimate_loss_and_accuracy(model, train_data, test_data)
            print(f"Step {iter}: Train Loss: {metrics['train']['loss']:.4f}, "
                  f"Train Acc: {metrics['train']['accuracy']:.2f}%, "
                  f"Val Loss: {metrics['val']['loss']:.4f}, "
                  f"Val Acc: {metrics['val']['accuracy']:.2f}%")

            if metrics['val']['accuracy'] > best_val_acc:
                best_val_acc = metrics['val']['accuracy']
                best_model_path = os.path.join(model_save_dir, "HeartGPT_best.pth")
                torch.save({
                    'iteration': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                }, best_model_path)
                print(f"New best model saved! Validation Accuracy: {best_val_acc:.2f}%")

        xb, yb = get_batch('train', train_data, test_data, current_batch_idx)
        logits, loss = model(xb, yb)

        with torch.no_grad():
            logits_reshaped = logits.view(-1, logits.size(-1))
            yb_reshaped = yb.view(-1)
            predictions = logits_reshaped.argmax(dim=-1)
            batch_accuracy = (predictions == yb_reshaped).float().mean().item() * 100

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(
                f"Iteration {iter}, Global Batch {global_batch_idx}, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")

        global_batch_idx += 1

if __name__ == "__main__":
    data_dir = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset"
    model_save_dir = r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\Model_files"

    os.makedirs(model_save_dir, exist_ok=True)

    train_model(data_dir, model_save_dir)