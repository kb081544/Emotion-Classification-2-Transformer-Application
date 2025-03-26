'''
high_peak_negative_chunks.csv 파일을 읽어서 각 행을 컨텍스트로 사용하여
HeartGPT로 새로운 PPG 데이터를 생성하는 스크립트
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# HeartGPT 모델 설정
model_config = 'PPG_PT'  # PPG 데이터 사용
block_size = 500  # 컨텍스트 길이
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
vocab_size = 102  # PPGPT용

model_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\Model_files\PPGPT_500k_iters\PPGPT_500k_iters.pth"

input_csv_path = "high_peak_negative_chunks.csv"  # 증강할 원본 데이터 파일
output_dir = "augmented_ppg_data"  # 생성된 데이터를 저장할 디렉토리

os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def tokenize_biosignal(data):
    shape = data.shape
    if len(shape) > 1 and shape[0] > shape[1]:
        data = data.T

    if data.shape[1] > 500:
        data = data[:, -500:]

    data_min = np.min(data)
    data_max = np.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)
    data_scaled *= 100
    data_rounded = np.round(data_scaled)

    return data_rounded, data_min, data_max

def inverse_tokenize(tokens, data_min, data_max):
    tokens_array = np.array(tokens)
    tokens_array = tokens_array / 100.0
    original_scale = tokens_array * (data_max - data_min) + data_min

    return original_scale

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
        wei = q @ k.transpose(-2, -1) * C**-0.5
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

class Heart_GPT_Model(nn.Module):
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
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx는 현재 컨텍스트의 인덱스가 포함된 (B, T) 배열
        for _ in range(max_new_tokens):
            # block_size 길이에 맞게 idx(컨텍스트) 자르기
            idx_cond = idx[:, -block_size:]
            # 예측 얻기
            logits, loss = self(idx_cond)
            # 마지막 시간 스텝에만 집중
            logits = logits[:, -1, :]  # (B, C) 형태로 변환
            # softmax 적용하여 확률 얻기
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 분포에서 샘플링
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 샘플링된 인덱스를 시퀀스에 추가
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# 모델 로드
print("모델 로딩 중...")
model = Heart_GPT_Model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)
print("모델 로딩 완료")

# 입력 CSV 파일 로드
print(f"입력 파일 로드 중: {input_csv_path}")
try:
    df = pd.read_csv(input_csv_path, header=None)
    data_chunks = df.values
    print(f"로드된 청크 수: {data_chunks.shape[0]}")
except Exception as e:
    print(f"파일 로드 중 오류 발생: {str(e)}")
    exit(1)

print("원본 데이터의 전체 범위 계산 중...")
all_data = data_chunks.flatten()
global_min = np.min(all_data)
global_max = np.max(all_data)
print(f"원본 데이터 전체 범위: min={global_min}, max={global_max}")

print("데이터 증강 시작...")
augmented_data_list = []
original_data_list = []
metadata_list = []

for chunk_idx in tqdm(range(data_chunks.shape[0]), desc="청크 처리 중"):
    chunk = data_chunks[chunk_idx:chunk_idx+1, :]

    chunk_min = np.min(chunk)
    chunk_max = np.max(chunk)

    tokenized_chunk, data_min, data_max = tokenize_biosignal(chunk)
    context_tensor = torch.tensor(tokenized_chunk, dtype=torch.long, device=device)

    original_data_list.append(chunk[0])

    num_samples_per_chunk = 3
    for i in range(num_samples_per_chunk):
        output_tokens = model.generate(context_tensor, max_new_tokens=block_size)[0].cpu().numpy()

        augmented_signal = inverse_tokenize(output_tokens, chunk_min, chunk_max)
        augmented_data_list.append(augmented_signal)
        metadata_list.append({
            'chunk_idx': chunk_idx,
            'sample_idx': i,
            'min_value': chunk_min,
            'max_value': chunk_max
        })

        sample_df = pd.DataFrame(augmented_signal)
        sample_path = os.path.join(output_dir, f"augmented_chunk_{chunk_idx}_sample_{i}.csv")
        sample_df.to_csv(sample_path, index=False, header=False)

        token_df = pd.DataFrame(output_tokens)
        token_path = os.path.join(output_dir, f"tokens_chunk_{chunk_idx}_sample_{i}.csv")
        token_df.to_csv(token_path, index=False, header=False)

augmented_data_array = np.array(augmented_data_list)
augmented_df = pd.DataFrame(augmented_data_array)
augmented_df.to_csv(os.path.join(output_dir, "all_augmented_data.csv"), index=False, header=False)

original_data_array = np.array(original_data_list)
original_df = pd.DataFrame(original_data_array)
original_df.to_csv(os.path.join(output_dir, "all_original_data.csv"), index=False, header=False)

metadata_df = pd.DataFrame(metadata_list)
metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print(f"데이터 증강 완료, 생성된 총 샘플 수: {len(augmented_data_list)}")
print(f"증강된 데이터는 {output_dir} 에 저장됨")
print(f"각 샘플의 원본 정보는 metadata.csv 파일에 저장됨")