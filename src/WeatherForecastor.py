import torch 
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from plotting import plot_difference

seed = 1
torch.manual_seed(seed)

dataset_path = './data/weatherHistory.csv'
df = pd.read_csv(dataset_path)
univariate_df = df['Temperature (C)']
univariate_df.index = df['Formatted Date']

input_size = 6
label_size = 1
offset = 1

def slicing_window(df, df_start_idx, df_end_idx, input_size, label_size, offset):
    X, y = [], []
    window_size = input_size + offset
    if df_end_idx is None:
        df_end_idx = len(df) - window_size
    for i in range(df_start_idx, df_end_idx):
        X.append(df[i:i+input_size])
        y.append(df[i+window_size-label_size:i+window_size])
    return np.expand_dims(np.array(X), -1), np.array(y)

dataset_len = len(univariate_df)
train_size = 0.7
val_size = 0.2
train_end_idx = int(dataset_len * train_size)
val_end_idx = int(dataset_len * (train_size + val_size))

X_train, y_train = slicing_window(univariate_df, 0, train_end_idx, input_size, label_size, offset)
X_val, y_val = slicing_window(univariate_df, train_end_idx, val_end_idx, input_size, label_size, offset)
X_test, y_test = slicing_window(univariate_df, val_end_idx, None, input_size, label_size, offset)

class WeatherForecast(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

train_dataset = WeatherForecast(X_train, y_train)
val_dataset = WeatherForecast(X_val, y_val)
test_dataset = WeatherForecast(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class WeatherForecastModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, dropout_prob):
        super().__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, label_size)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.norm(x[:, -1, :])
        x = self.dropout(x)
        return self.fc(x)

def fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        batch_train_losses = []
        for text, label in train_loader:
            text, label = text.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_losses.append(np.mean(batch_train_losses))
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}')
    return train_losses, val_losses

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_losses, outputs, y_test = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_losses.append(loss.item())
            outputs.append(output.cpu().numpy())
            y_test.append(y.cpu().numpy())
    return np.mean(val_losses), outputs, y_test

if __name__ == "__main__":
    embedding_dim = 1
    hidden_size = 8
    n_layers = 3
    dropout_prob = 0.2
    lr = 1e-3
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WeatherForecastModel(embedding_dim, hidden_size, n_layers, dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = fit(model, train_loader, val_loader, criterion, optimizer, device, epochs)
    val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    test_loss, outputs, y_test = evaluate(model, test_loader, criterion, device)

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

    os.makedirs('./results', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_loss': val_loss,
        'test_loss': test_loss
    }
    with open('./results/problem2.json', 'w') as f:
        json.dump(results, f)

    flatten_y = np.concatenate(y_test).flatten()
    flatten_pred = np.concatenate(outputs).flatten()
    plot_difference(flatten_y[:100], flatten_pred[:100])

    torch.save(model.state_dict(), './models/problem2.pth')
