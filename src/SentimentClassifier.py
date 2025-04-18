import torch 
import torch.nn as nn

seed = 1
torch.manual_seed(seed)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')

import unidecode
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from plotting import plotting

import json
import pickle

dataset_path = './data/all-data.csv'
headers = ['sentiment', 'content']
df = pd.read_csv(dataset_path, names=headers, encoding='ISO-8859-1')

classes = {class_name: idx for idx, class_name in enumerate(df['sentiment'].unique().tolist())}
df['sentiment'] = df['sentiment'].apply(lambda x: classes[x])

english_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

def text_normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split(' ') if word not in english_stopwords])
    text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    return text

df['content'] = df['content'].apply(lambda x: text_normalize(x))

vocab = []
for sentence in df['content'].tolist():
    tokens = sentence.split()
    for token in tokens:
        if token not in vocab:
            vocab.append(token)
            
vocab.append('<pad>')
vocab.append('<unk>')
word2idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

def transform(text, word2idx, max_seq_len):
    tokens = []
    for w in text.split():
        try:
            w_ids = word2idx[w]
        except:
            w_ids = word2idx['<unk>']
        tokens.append(w_ids)
    if len(tokens) < max_seq_len:
        tokens += [word2idx['<pad>']] * (max_seq_len - len(tokens))
    else:
        tokens = tokens[:max_seq_len]
    return tokens

val_size = 0.2
test_size = 0.125
is_shuffle = True
texts = df['content'].tolist()
labels = df['sentiment'].tolist()

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=val_size, random_state=seed, shuffle=is_shuffle)
X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=seed, shuffle=is_shuffle)

max_seq_len = 32
train_batch_size = 128
test_batch_size = 8

class FinancialNews(Dataset):
    def __init__(self, X, y, word2idx, max_seq_len, transform=None):
        self.texts = X
        self.labels = y
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len
        self.transform = transform
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.transform:
            text = self.transform(text, self.word2idx, self.max_seq_len)
        text, label = torch.tensor(text), torch.tensor(label)
        return text, label
    
train_dataset = FinancialNews(X_train, y_train, word2idx, max_seq_len, transform=transform)
val_dataset = FinancialNews(X_val, y_val, word2idx, max_seq_len, transform=transform)
test_dataset = FinancialNews(X_test, y_test, word2idx, max_seq_len, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, n_classes, dropout_prob):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x, hn = self.rnn(x)
        x = self.norm(x[:, -1, :])
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
n_classes = len(list(classes.keys()))
embedding_dim = 64
hidden_size = 64
n_layers = 2
dropout_prob = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentClassifier(vocab_size, embedding_dim, hidden_size, n_layers, n_classes, dropout_prob).to(device)
lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epochs = 50

def fit(model, train_loader, val_loader, criterion, optimizer, devive, num_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        batch_train_losses = []
        for idx, (text, label) in enumerate(train_loader):
            text, label = text.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            batch_train_losses.append(loss.item())
            
        train_loss = np.mean(batch_train_losses)
        train_losses.append(train_loss)
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
    return train_losses, val_losses
        
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text, label in val_loader:
            text, label = text.to(device), label.to(device)
            
            output = model(text)
            loss = criterion(output, label)
            val_losses.append(loss.item())
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
    val_loss = np.mean(val_losses)
    val_acc = correct / total
    return val_loss, val_acc

if __name__ == "__main__":
    train_losses, val_losses = fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    with open('./results/problem1.json', 'w') as f:
        json.dump(results, f)
        
    plotting('./results/problem1.json')

    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(model.state_dict(), './models/problem1.pth')
    with open('./models/vocab.pkl', 'wb') as f:
        pickle.dump(word2idx, f)