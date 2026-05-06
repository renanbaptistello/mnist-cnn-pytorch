import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


n_epochs = 15
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random_seed = 1
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)


class CNN (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv2= nn.Conv2d(64, 128, kernel_size=3)
        self.dropout2 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) 
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    

def train(network, train_loader, optimizer, loss_fn, epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # if batch_idx % log_interval == 0:
        #     print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")


def test(network, test_loader, loss_fn):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = network(data)
            loss = loss_fn(output, target)

            test_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')
    

def validate(network, val_loader, loss_fn, epoch):
    network.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = network(data)
            loss = loss_fn(output, target)

            val_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1} -> Validation: Avg loss: {val_loss:.4f}, '
          f'Accuracy: {correct}/{len(val_loader.dataset)} '
          f'({100. * correct / len(val_loader.dataset):.2f}%)')

def main():
    network = CNN().to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    full_train_dataset = torchvision.datasets.MNIST(
        'data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_test, shuffle=False)
    

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('data', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size_test, shuffle=False)

    for epoch in range(n_epochs):
        train(network, train_loader, optimizer, loss_fn, epoch)
        validate(network, val_loader, loss_fn, epoch)

    test(network, test_loader, loss_fn)


if __name__ == "__main__":
    main()