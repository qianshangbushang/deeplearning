import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.optim  as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(input_size * hidden_size, 10)
        return
    
    def forward(self, x, h):
        out, h_out = self.rnn(x, h)
        return self.fc(out.reshape(x.shape[0], -1))
    


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        return
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

def train():
    train_dataset = datasets.MNIST('dataset/minist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST('dataset/minist', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


    model = RNN(28, 56, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    h = torch.rand((3, 32, 56))
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device).squeeze(1)
            target = target.to(device)


            logits = model(data, None)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item()}")
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


def check_accuracy(loader:DataLoader, model:nn.Module):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x, None)
            _, preds = scores.max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)* 100}" )
    model.train()
    return

if __name__ == "__main__":
    # x: (16, 28) (28, 56) -> (16, 56)
    # h: (16, 56) (56, 56) -> (16, 56)
    # x = torch.rand((16, 28, 28))
    # h = torch.rand((3, 16, 56))
    # rnn = RNN(28, 56, 3)
    # print(rnn(x, h).shape)
    train()
