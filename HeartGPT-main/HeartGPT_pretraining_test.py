import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Model configuration
block_size = 500
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
vocab_size = 101
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main_example_dataset\HeartGPT-main\Model_files\PPGPT_pretrained_64_8_8_500_1000000_500000_limit_amplitude.pth"


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
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
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    # Load data
    data = np.loadtxt(r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main_example_dataset\HeartGPT-main\filtered_processed_blocksize_500_strides_500_ppg_data_limit_amplitude.txt", delimiter=",")

    sequence_idx = np.random.randint(0, data.shape[0] - 1)
    input_sequence = data[sequence_idx]
    next_sequence = data[sequence_idx + 1]

    model = HeartGPTModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    '''
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_sequence = model.generate(input_tensor, max_new_tokens=block_size)
    predicted_block = predicted_sequence[0, block_size:].cpu().numpy()

    plt.figure(figsize=(15, 5))

    x_input = np.arange(250)
    x_next = np.arange(250, 500)
    plt.plot(x_input, input_sequence, 'k-', label='Input Sequence')
    plt.plot(x_next, next_sequence, 'k-', label='Next Sequence')

    plt.plot(x_next, predicted_block, 'r-', label='Predicted Sequence', alpha=0.7)

    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('HeartGPT Prediction vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    output_dir = r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\prediction_plots\example_dataset"
    '''
    for i in range(100, 130):  # Sequences from 100 to 150
        input_sequence = data[i]
        next_sequence = data[i + 1]

        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_sequence = model.generate(input_tensor, max_new_tokens=block_size)
        predicted_block = predicted_sequence[0, block_size:].cpu().numpy()

        plt.figure(figsize=(15, 5))

        x_input = np.arange(block_size)
        x_next = np.arange(block_size, block_size+block_size)
        plt.plot(x_input, input_sequence, 'k-', label='Input Sequence')
        plt.plot(x_next, next_sequence, 'k-', label='Next Sequence')
        plt.plot(x_next, predicted_block, 'r-', label='Predicted Sequence', alpha=0.7)

        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title(f'HeartGPT Prediction vs Actual (Sequence {i})')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, f'filtered_sequence_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Saved sequence {i}')
        '''
    for i in range(109, 130):  # Sequences from 100 to 150
        input_sequence = data[i]
        next_sequence = data[i + 1]

        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_sequence = model.generate(input_tensor, max_new_tokens=block_size)
        predicted_block = predicted_sequence[0, block_size:].cpu().numpy()

        plt.figure(figsize=(15, 5))

        x_next = np.arange(block_size, block_size+block_size)

    #     print(f'Saved sequence {i}')
    #
    # for i in range(data.shape[0] - 1):
    #     input_sequence = data[i]
    #     next_sequence = data[i + 1]
    #
    #     input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
    #
    #     with torch.no_grad():
    #         predicted_sequence = model.generate(input_tensor, max_new_tokens=block_size)
    #     predicted_block = predicted_sequence[0, block_size:].cpu().numpy()
    #
    #     plt.figure(figsize=(15, 5))

        x_input = np.arange(block_size)
        x_next = np.arange(block_size, block_size+block_size)
        plt.plot(x_input, input_sequence, 'k-', label='Input Sequence')
        plt.plot(x_next, predicted_block, 'r-', label='Predicted Sequence', alpha=0.7)

        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title(f'HeartGPT Prediction vs Actual (Sequence {i + 1}/{data.shape[0] - 1})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'filtered_sequence_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()