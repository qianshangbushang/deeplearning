
import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from rnn import RNN


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers=3, dropout=0.1, bi_directional=False) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_directional = bi_directional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bi_directional)
        self.fc = nn.Linear(input_size * hidden_size, num_class)
        return
    
    def forward(self, x:torch.Tensor):
        d = 2 if self.bi_directional else 1
        h0 = torch.zeros((d * self.num_layers, x.shape[0], self.hidden_size))
        c0 = torch.zeros((d * self.num_layers, x.shape[0], self.hidden_size))
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return self.fc(out.reshape(x.shape[0], -1))
        

def create_dataloader(mode='train', batch_size=32, shuffle=True):
    dataset = datasets.MNIST(
        "./dataset/minist",
        train=(mode == 'train'),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambd=lambda x: x.squeeze(0))
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def check_accuracy(model: nn.Module, dataloader: DataLoader, device: str):
    num_correct, num_samples = 0, 0
    model.eval()
    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            _, preds = logits.max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    model.train()
    return float(num_correct) / float(num_samples)


def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, max_epoch=3, device=torch.device('cpu'), lr=0.001):
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(max_epoch):
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 or batch_idx == len(train_dataloader) - 1:
                print(f"Epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}")
        train_acc = check_accuracy(model, train_dataloader, device)
        test_acc = check_accuracy(model, test_dataloader, device)

        print(
            f"Epoch: {epoch} train accuracy: {train_acc}, test accuracy: {test_acc}")
    return

def save_checkpoint(model:nn.Module, optimizer: optim.Optimizer, filepath:str):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(ckpt, filepath)

def run_rnn():
    model = RNN(28, 56, 3)
    train(
        model, 
        train_dataloader=create_dataloader('train'), 
        test_dataloader=create_dataloader('test'),
    )
    print("finished")

def run_lstm():
    model = LSTM(28, 56, 10)
    train(
        model, 
        train_dataloader=create_dataloader('train'), 
        test_dataloader=create_dataloader('test'),
    )
    print("finished")


def test_lstm():
    model = LSTM(28, 56, 10)
    x = torch.rand((10, 28, 28))
    print(model(x).shape)
    
if __name__ == "__main__":
    # run()
    run_lstm()