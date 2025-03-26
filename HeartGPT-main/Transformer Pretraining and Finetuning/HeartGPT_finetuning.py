import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

# Harry Davies 19_09_2024

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775


model_config = 'PPG_PT' #switch between 'ECG_PT' and 'PPG_PT'

eval_interval = 50
save_interval = 20000
max_iters = 2000
eval_iters = 50
batch_size = 128
block_size = 500
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
learning_rate = 3e-04
model_path_ppg = r"/HeartGPT-main_example_dataset/HeartGPT-main/Model_files/PPGPT_500k_iters/PPGPT_500k_iters.pth"
model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"

model_path_finetune = r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\Model_files\finetune_classifiction.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102 #102 for PPGPT, 101 for ECGPT
    model_path = model_path_ppg
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg

# load in the data, in our case data was originally prepared in matlab
# original fine tuning data for AFib had training data (X) dimensions of Nx500, and label dimensions (Y) of Nx2. 
# For afib pne of the labels in Y was subject number, and used to exclude subjects during cross-validation. The first of the 2 values was the AF class of 0 or 1.
# For beat detection fine tuning, training data (X) had dimensions of Nx500, and label dimenions (Y) of Nx500, where 0 corresponded to no beat, and 1 was labelled at the position of a beat.
data_load_positive=np.loadtxt(r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\filtered_processed_blocksize_500_strides_500_ppg_data_limit_amplitude_positive_normalized.txt", delimiter=',')
data_load_negative=np.loadtxt(r"C:\Users\user\PycharmProjects\Emotion Classification 3\HeartGPT-main\filtered_processed_blocksize_500_strides_500_ppg_data_limit_amplitude_negative_normalized.txt", delimiter=',')

y_posivie=np.zeros((data_load_positive.shape[0],1))
y_negative=np.ones((data_load_negative.shape[0],1))

data=np.concatenate((data_load_positive, data_load_negative), axis=0)
data_y=np.concatenate((y_posivie, y_negative), axis=0)



# Get the permutation of indices
perm = np.random.permutation(data.shape[0])

# Shuffle X and y
X_shuffled = data[perm]
y_shuffled = data_y[perm]

# split into train and test
# trainX, testX, trainy, testy = train_test_split(X_shuffled, y_shuffled, test_size=0.1, random_state=10)


def get_batch(X, y, batch_size):
    ix = torch.randint(len(X), (batch_size,))
    x_tensor = torch.tensor(X, dtype=torch.int64)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    x = x_tensor[ix]
    y = y_tensor[ix].squeeze()

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, X_train, y_train, X_val, y_val):
    model.eval()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    criterion = nn.BCELoss()

    # Calculate training metrics
    for _ in range(eval_iters):
        xb, yb = get_batch(X_train, y_train, batch_size)
        logits = model(xb)
        loss = criterion(logits, yb)
        train_losses.append(loss.item())

        predictions = (logits >= 0.5).float()
        accuracy = (predictions == yb).float().mean()
        train_accuracies.append(accuracy.item())

    # Calculate validation metrics
    for _ in range(eval_iters):
        xb, yb = get_batch(X_val, y_val, batch_size)
        logits = model(xb)
        loss = criterion(logits, yb)
        val_losses.append(loss.item())

        # Calculate accuracy
        predictions = (logits >= 0.5).float()
        accuracy = (predictions == yb).float().mean()
        val_accuracies.append(accuracy.item())

    model.train()
    return (np.mean(train_losses), np.mean(val_losses),
            np.mean(train_accuracies), np.mean(val_accuracies))

#model definition
class Head(nn.Module):

    def __init__(self, head_size, mask=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.mask = mask
        self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        if self.mask:
            wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
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

class NewHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n_embd))
        self.bias = nn.Parameter(torch.zeros(1))
        self.SigM1 = nn.Sigmoid()

    def forward(self, x):
        x = x[:,-1,:]
        x = F.linear(x, self.weight)
        x = self.SigM1(x)
        return x.squeeze()

class Heart_GPT_FineTune(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer - 1)] +
                                  [Block(n_embd, n_head=n_head, mask=True)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = None

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def train_model(X, y, model_path):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = Heart_GPT_FineTune()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if not k.startswith('lm_head')}
    model.load_state_dict(pretrained_dict, strict=False)
    model.lm_head = NewHead(n_embd)

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.ln_f.parameters():
        param.requires_grad = True
    for param in model.blocks[-1].parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.BCELoss()

    for iter in range(max_iters):
        xb, yb = get_batch(X_train, y_train, batch_size)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0:
            train_loss, val_loss, train_acc, val_acc = estimate_loss(model, X_train, y_train, X_val, y_val)
            print(f'step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, '
                  f'train acc {train_acc:.4f}, val acc {val_acc:.4f}')
    return model


model = train_model(X_shuffled, y_shuffled, model_path_ppg)
torch.save(model.state_dict(), 'finetuned_model.pth')


