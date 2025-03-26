import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import heartpy as hp
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


eval_interval = 2000
save_interval = 20000
eval_iters = 200
batch_size = 8
max_iters = 1000000
block_size = 250
learning_rate = 3e-04
n_embd = 16
n_head = 4
n_layer = 4
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_ppg_signal(valid_data):

    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    normalized_signal = ((valid_data - min_val) / (max_val - min_val)) * 100

    return normalized_signal


def save_matrix_to_txt(matrix, filename):
    try:
        np.savetxt(filename, matrix, fmt='%d', delimiter=',')
        print(f"Successfully saved matrix to {filename}")
        print(f"Matrix shape: {matrix.shape}")
    except Exception as e:
        print(f"Error saving matrix: {str(e)}")


def reshape_vector_to_matrix(vector, row_size=block_size):
    n = len(vector)
    stride = row_size // 1

    valid_segments = []

    i = 0
    while i * stride + row_size <= n:
        start_idx = i * stride
        end_idx = min(start_idx + row_size, n)
        window_data = vector[start_idx:end_idx]

        if len(window_data) < row_size:
            window_data = np.pad(window_data, (0, row_size - len(window_data)),
                                 mode='constant', constant_values=np.nan)

        valid_data = window_data[~np.isnan(window_data)]
        valid_data = hp.filter_signal(valid_data, cutoff=[0.5, 8],
                                      sample_rate=25, order=3, filtertype="bandpass")

        try:
            wd, m = hp.process(valid_data, sample_rate=25)
            if (len(wd['peaklist']) - len(wd['removed_beats'])) > (block_size / 25) / 2:
                processed_signal = process_ppg_signal(valid_data)
                valid_segments.append(processed_signal[:row_size])

        except Exception as e:
            print(f"세그먼트 {i} 처리 실패: {str(e)}")
        i += 1

    if not valid_segments:
        return None

    num_valid_segments = len(valid_segments)
    matrix = np.zeros((num_valid_segments, row_size))

    for i, segment in enumerate(valid_segments):
        if len(segment) < row_size:
            segment = np.pad(segment, (0, row_size - len(segment)),
                             mode='constant', constant_values=0)
        matrix[i, :] = segment

    matrix = np.round(matrix).astype(np.int64)
    matrix = np.clip(matrix, 0, 100)

    output_filename = f"filtered_processed_blocksize_{block_size}_strides_{stride}.txt"
    save_matrix_to_txt(matrix, output_filename)

    return matrix

file_dir=r"concatenated_ppg_data.txt"
data_load = np.loadtxt(file_dir)
data_reshaped=reshape_vector_to_matrix(data_load)

vocab_size = 101

perm = np.random.permutation(data_reshaped.shape[0])
data_ppg_rand = data_reshaped[perm, :]

data = torch.tensor(data_ppg_rand, dtype=torch.long)

x_thresh = int(0.9 * data_reshaped.shape[0])
train_data = data[:x_thresh, :]
test_data = data[x_thresh:, :]


def get_batch(split):
    data = train_data if split == 'train' else test_data

    ix = torch.randint(data.shape[0], (batch_size,))

    ix2 = torch.randint(0, data.shape[1] - block_size + 1, (1,))

    x = torch.stack([data[i, ix2:ix2 + block_size - 1] for i in ix])
    y = torch.stack([data[i, ix2 + 1:ix2 + block_size] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss_and_accuracy():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            accuracies[k] = calculate_accuracy(logits.view(-1, logits.size(-1)), Y.view(-1))
        out[split] = {
            'loss': losses.mean().item(),
            'accuracy': accuracies.mean().item() * 100
        }
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',
                             torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # start = time.time()
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # all attention weights sum to 1 for updating a single token
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        # end = time.time()
        # print(start-end)
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
        # communication
        self.sa = MultiHeadAttention(n_head, head_size)
        # computation
        self.ffwd = FeedForward(n_embd)
        # layer norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class HeartGPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T, C
        x = tok_emb + pos_emb  # B, T, C
        x = self.blocks(x)  # B, T, C
        x = self.ln_f(x)  # B, T, C

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = HeartGPTModel()
m = model.to(device)

# AdamW
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

def calculate_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_parameters = count_parameters(model)
print(f"The model has {num_parameters} trainable parameters.")

for iter in range(max_iters):

    if iter % eval_interval == 0:
        metrics = estimate_loss_and_accuracy()
        print(f"step {iter}: train loss {metrics['train']['loss']:.4f}, val loss {metrics['val']['loss']:.4f}")
        print(f"          train acc {metrics['train']['accuracy']:.2f}%, val acc {metrics['val']['accuracy']:.2f}%")
    if iter % save_interval == 0:
        model_path = f"C:/Users/user/PycharmProjects/Emotion Classification 3/HeartGPT-main/Model_files/PPGPT_pretrained_{n_embd}_{n_head}_{n_layer}_{block_size}_{max_iters}_{iter}_limit_amplitude.pth"
        torch.save(model.state_dict(), model_path)
    x_batch, y_batch = get_batch('train')

    logits, loss = m(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()






