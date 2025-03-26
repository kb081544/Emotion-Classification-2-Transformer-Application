import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
'''
추론 모델이 1이라고 하는 것과 0이라고 하는 얘가 다른지
0.5는 애매한게 맞는데, 

13 개 중에 앞부분과 뒷 부분의 파형 자체가 다른지 확인해봐야 함 
0에 해당되는 데이터와 1에 해당되는 데이터가 달라야 하는데 처음부터 다르지 않으면 데이터 자체가 잘못된거임
심박수라도 뽑아보자 가르고자 하는 데이터가 가를만 한 데이터인지
전반부랑 후반부를 봤을 때 데이터의 파형이 의미적으로 다르다면, 눈으로 보인다면 

정 안되면 스트레스 지수를 기준으로 해서 

정답을 0 이라고 생각했던 청크와 1이라고 생각했던 청크가 다 맞는지 확인하는 것
레이블링을 제대로 한 것이 맞나? 데이터를 재수집을 해야 하나

스트레스 sdk
'''
# Configuration parameters (kept from the original script)
block_size = 500
n_embd = 64
n_head = 8
n_layer = 8
vocab_size = 102  # for PPGPT
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Recreate the model classes from the original script
class NewHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n_embd))
        self.bias = nn.Parameter(torch.zeros(1))
        self.SigM1 = nn.Sigmoid()

    def forward(self, x):
        x = x[:, -1, :]
        x = F.linear(x, self.weight)
        x = self.SigM1(x)
        return x.squeeze()


class Head(nn.Module):
    def __init__(self, head_size, mask=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.mask = mask
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, mask=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, mask=mask) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)

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
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, mask=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, mask=mask)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Heart_GPT_FineTune(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer - 1)] +
                                     [Block(n_embd, n_head=n_head, mask=True)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = NewHead(n_embd)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def load_model(model_path):
    model = Heart_GPT_FineTune()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict(model, test_data):
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_data, dtype=torch.int64).to(device)
        predictions = model(test_tensor)
        predictions = predictions.cpu().numpy()
    return predictions

def analyze_single_file(model, file_path):
    try:
        test_data = np.loadtxt(file_path, delimiter=',')

        total_samples = test_data.shape[0]
        mid_point = total_samples // 2

        ground_truth = np.zeros(total_samples)
        ground_truth[mid_point:] = 1

        test_data_x = test_data[:, 1:]

        predictions = predict(model, test_data_x)
        #for i in range(len(test_data_x)):

        first_half_predictions = predictions[:mid_point]
        first_half_ground_truth = ground_truth[:mid_point]

        second_half_predictions = predictions[mid_point:]
        second_half_ground_truth = ground_truth[mid_point:]

        first_half_accuracy = np.mean(first_half_predictions < 0.5)
        second_half_accuracy = np.mean(second_half_predictions >= 0.5)

        return {
            'filename': os.path.basename(file_path),
            'first_half_mean_prediction': np.mean(first_half_predictions),
            'second_half_mean_prediction': np.mean(second_half_predictions),
            'first_half_accuracy': first_half_accuracy,
            'second_half_accuracy': second_half_accuracy,
            'total_samples': total_samples,
            'mid_point': mid_point
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def analyze_directory(model_path, directory_path):
    model = load_model(model_path)

    test_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]

    results = []
    for file_path in test_files:
        result = analyze_single_file(model, file_path)
        if result:
            results.append(result)

    return results


if __name__ == "__main__":
    model_path = 'finetuned_model.pth'
    test_data_directory = r"C:\Users\user\PycharmProjects\Emotion Classification 3\Dataset\ppgDataset\processed_test_data"

    analysis_results = analyze_directory(model_path, test_data_directory)

    print("Analysis Results:")
    for result in analysis_results:
        print(f"\nFile: {result['filename']}")
        print(f"Total Samples: {result['total_samples']} (Mid-point: {result['mid_point']})")
        print(f"First Half (Expected Class 0):")
        print(f"  - Mean Prediction: {result['first_half_mean_prediction']:.4f}")
        print(f"  - Accuracy: {result['first_half_accuracy']:.4f}")
        print(f"Second Half (Expected Class 1):")
        print(f"  - Mean Prediction: {result['second_half_mean_prediction']:.4f}")
        print(f"  - Accuracy: {result['second_half_accuracy']:.4f}")

