import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


def train(model,
          train_loader,
          optimizer,
          device):
    model.train()
    for iter_, (x, t) in enumerate(train_loader):
        x = x.to(device)
        t = t.to(device)
        optimizer.zero_grad()
        y = model(x)
        loss = nn.CrossEntropyLoss()(y, t)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(model(x), dim=1)
        acc = pred.eq(t.view_as(pred)).sum().item() / x.shape[0]
        print('loss: %.3f  acc: %.3f' % (loss, acc), end='\r')


def test(model,
         test_loader,
         device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for iter_, (x, t) in enumerate(test_loader):
            x = x.to(device)
            t = t.to(device)
            pred = torch.argmax(model(x), dim=1)
            correct += pred.eq(t.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print('Accuracy: %.3f' % acc)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', '-bs',
                        type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--nb_epoch', '-e',
                        type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', '-lr',
                        type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', '-m',
                        type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda',
                        dest='use_cuda',
                        action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)

    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum)

    for epoch in range(1, args.nb_epoch+1):
        print('epoch %d / %d' % (epoch, args.nb_epoch))
        train(model, train_loader, optimizer, device)
        print()

    test(model, test_loader, device)


if __name__ == '__main__':
    main()
