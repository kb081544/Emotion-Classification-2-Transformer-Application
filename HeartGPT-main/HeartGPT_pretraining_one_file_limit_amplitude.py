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

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

eval_interval = 2000
save_interval = 20000  # how often the model is checkpointed
eval_iters = 200
batch_size = 8  # sequences we process in parellel
max_iters = 1000000
block_size = 500  # this is context length
# 원본은 50Hz의 10초 즉 500 블록.
learning_rate = 3e-04
n_embd = 16  # 384 / 6 means every head is 64 dimensional
n_head = 4
n_layer = 4
dropout = 0.2

# GPU is necessary. Training of 8 head, 8 layer model and 500 context length was possible with 12GB VRAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def normalize_peak_heights(signal):
#     peaks, _ = scipy.signal.find_peaks(signal)
#
#     if len(peaks) == 0:
#         return signal
#
#     peak_heights = signal[peaks]
#     median_height = np.median(peak_heights)
#
#     normalized_signal = signal.copy()
#     for peak in peaks:
#         current_height = signal[peak]
#         if current_height > median_height:
#             window_start = max(0, peak - 5)  # 조정 가능한 윈도우 크기
#             window_end = min(len(signal), peak + 6)
#             window = normalized_signal[window_start:window_end]
#             scaling_factor = median_height / current_height
#             normalized_signal[window_start:window_end] = window * scaling_factor
#
#     return normalized_signal
#
#
# def limit_amplitude_variation(signal):
#     """
#     PPG 신호의 진폭을 제한하고 정규화하는 함수
#     """
#     peaks, _ = scipy.signal.find_peaks(signal, distance=10)
#     valleys, _ = scipy.signal.find_peaks(-signal, distance=10)
#
#     if len(peaks) == 0 or len(valleys) == 0:
#         return signal
#
#     modified_signal = signal.copy()
#
#     peak_heights = signal[peaks]
#     valley_depths = signal[valleys]
#
#     peak_mean = np.mean(peak_heights)
#     valley_mean = np.mean(valley_depths)
#
#     desired_peak_height = peak_mean
#     desired_valley_depth = valley_mean
#
#     for i in range(len(peaks) - 1):
#         start_idx = peaks[i]
#         end_idx = peaks[i + 1]
#
#         segment = modified_signal[start_idx:end_idx]
#
#         seg_max = np.max(segment)
#         seg_min = np.min(segment)
#
#         if seg_max != seg_min:
#             normalized_segment = (segment - seg_min) / (seg_max - seg_min)
#             scaled_segment = normalized_segment * (desired_peak_height - desired_valley_depth) + desired_valley_depth
#             modified_signal[start_idx:end_idx] = scaled_segment
#
#     return modified_signal


def process_ppg_signal(processed_signal):

    # boundary_size = 20
    # valid_data[:boundary_size] = np.mean(valid_data[boundary_size:boundary_size * 2])
    # valid_data[-boundary_size:] = np.mean(valid_data[-boundary_size * 2:-boundary_size])
    #
    # processed_signal = limit_amplitude_variation(valid_data)

    min_val = np.min(processed_signal)
    max_val = np.max(processed_signal)
    normalized_signal = ((processed_signal - min_val) / (max_val - min_val)) * 100

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

    # 결과 저장
    output_filename = f"filtered_processed_blocksize_{block_size}_strides_{stride}_ppg_data_limit_amplitude.txt"
    save_matrix_to_txt(matrix, output_filename)

    return matrix

file_dir=r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\concated_ppg_data.txt"
data_load = np.loadtxt(file_dir)
data_reshaped=reshape_vector_to_matrix(data_load)

vocab_size = 101

perm = np.random.permutation(data_reshaped.shape[0])
data_ppg_rand = data_reshaped[perm, :]

# now time for some pytorch, convert to a torch tensor
data = torch.tensor(data_ppg_rand, dtype=torch.long)

# split so 90% for training, 10% for testing
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
            'accuracy': accuracies.mean().item() * 100  # Convert to percentage
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
                             torch.tril(torch.ones((block_size, block_size))))  # buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # start = time.time()
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # the tril signifies a decoder block, future tokens cannot communicate with the past
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
        # creating a list of head objects (turned into modules) resulting in a number of head modules
        # then assigns the list of modules to self.heads - these run in parellel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projection generally matches sizes for adding in residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate the output of the different attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # multiplication performed in attention is all you need paper
            # expands and contracts back down to projection
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


# create heart GPT class
class HeartGPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx is batch, targets is time
        tok_emb = self.token_embedding_table(idx)  # (B, T, vocab_size) which is batch, time, channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T, C (integers from 0 to T-1)
        x = tok_emb + pos_emb  # B, T, C
        x = self.blocks(x)  # B, T, C
        x = self.ln_f(x)  # B, T, C

        logits = self.lm_head(x)
        # channel is vocab size, so in this case 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx (context) to the last block_size tokens because positional embeddings only has up to block size
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = HeartGPTModel()
m = model.to(device)
# random loss at this point would be -log(1/65)

# AdamW
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

def calculate_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# counter the number of model parameters to be trained
num_parameters = count_parameters(model)
print(f"The model has {num_parameters} trainable parameters.")

for iter in range(max_iters):

    if iter % eval_interval == 0:
        metrics = estimate_loss_and_accuracy()
        print(f"step {iter}: train loss {metrics['train']['loss']:.4f}, val loss {metrics['val']['loss']:.4f}")
        print(f"          train acc {metrics['train']['accuracy']:.2f}%, val acc {metrics['val']['accuracy']:.2f}%")
    if iter % save_interval == 0:
        # model_path for checkpointing
        model_path = f"C:/Users/user/PycharmProjects/Emotion Classification 3/HeartGPT-main/Model_files/PPGPT_pretrained_{n_embd}_{n_head}_{n_layer}_{block_size}_{max_iters}_{iter}_limit_amplitude.pth"
        torch.save(model.state_dict(), model_path)
    # get batch
    x_batch, y_batch = get_batch('train')

    # loss evaluation
    logits, loss = m(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()






