import torch
import torch.nn as nn
import torch.optim as optim

class SimpleSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_svm(X_train, y_train, epochs=100, lr=0.01):
    # Assume binário para simplicidade; expanda para multi-classe
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) * 2 - 1  # -1/1
    X_train = torch.tensor(X_train, dtype=torch.float32)
    model = SimpleSVM(X_train.shape[1])
    criterion = nn.HingeEmbeddingLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def predict_svm(model, X_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    return (model(X_test) > 0).int().squeeze().numpy()