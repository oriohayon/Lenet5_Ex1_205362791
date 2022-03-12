import numpy as np
from matplotlib import pyplot as plt
from torch import optim

from lenet import LeNet5, LeNet5DropOut, LeNet5WeightDecay, LeNet5BatchNorm
import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = FashionMNIST('./data/FashionMNIST', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
data_test = FashionMNIST('./data/FashionMNIST', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

criterion = nn.CrossEntropyLoss()


def train(net, optimizer):
    net.train()
    loss_lst, accuracy_lst, epoch_steps = [], [], []
    total_correct = 0.0

    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

        loss.backward()
        optimizer.step()

    accuracy = total_correct / len(data_train)

    return loss.detach().cpu().item(), accuracy.item()


def test(net):
    net.eval()
    total_correct = 0.0
    loss_lst, accuracy_lst, epoch_steps = [], [], []

    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        loss = criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    accuracy = total_correct / len(data_test)

    return loss.detach().cpu().item(), accuracy.item()


def train_and_test_lenet5(n_epoch, mode='regular'):

    train_loss_lst, train_accuracy_lst = [], []
    test_loss_lst, test_accuracy_lst = [], []

    lr = 0.1

    if mode == 'DropOut':
        net = LeNet5DropOut()
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif mode == 'WeightDecay':
        net = LeNet5()
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-2)
    elif mode == 'BatchNorm':
        net = LeNet5BatchNorm()
        optimizer = optim.SGD(net.parameters(), lr=lr)
    else:
        net = LeNet5()
        optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(n_epoch):
        train_loss, train_accuracy = train(net, optimizer)
        print('Train - Epoch %d, Loss: %f, accuracy: %f' % (epoch, train_loss, train_accuracy))

        test_loss, test_accuracy = test(net)
        print('Test - Epoch: %d, Loss: %f, Accuracy: %f' % (epoch, test_loss, test_accuracy))

        train_loss_lst.append(train_loss)
        train_accuracy_lst.append(train_accuracy)
        test_loss_lst.append(test_loss)
        test_accuracy_lst.append(test_accuracy)

    loss_fig, loss_ax = plt.subplots()
    loss_ax.plot(range(1, n_epoch+1), train_loss_lst, label='Train')
    loss_ax.plot(range(1, n_epoch+1), test_loss_lst, label='Test')
    loss_ax.legend(loc='upper right')
    loss_ax.set_title('Log Loss Vs #epoches using ' + mode)

    acc_fig, acc_ax = plt.subplots()
    acc_ax.plot(range(1, n_epoch+1), train_accuracy_lst, label='Train')
    acc_ax.plot(range(1, n_epoch+1), test_accuracy_lst, label='Test')
    acc_ax.legend(loc='upper left')
    acc_ax.set_title('Accuracy Vs #epoches using ' + mode)
    plt.show(block=True)


def main(n_epoch, mode):
    train_and_test_lenet5(n_epoch, mode)


if __name__ == '__main__':
    main(n_epoch=6, mode='WeightDecay')
