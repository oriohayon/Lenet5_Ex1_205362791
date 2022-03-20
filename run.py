import sys
import os
import time
import torch

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import optim
from lenet import LeNet5, LeNet5DropOut, LeNet5BatchNorm
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_train = FashionMNIST('./data/FashionMNIST', download=True,
                          transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
data_test = FashionMNIST('./data/FashionMNIST', train=False, download=True,
                         transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True)
data_test_loader = DataLoader(data_test)


def single_epoch_train(net, optimizer, lenet_mode, active_train=True):
    total_correct = 0.0

    criterion = nn.CrossEntropyLoss()
    net.train()

    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        if lenet_mode.lower() == 'dropout':
            net.eval()
            output = net(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            net.train()
        else:
            output = net(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        if active_train:
            loss.backward()
            optimizer.step()

    accuracy = total_correct / len(data_train)

    return net, loss.detach().cpu().item(), accuracy.item()


def train_lenet(lenet_mode='Regular', n_epoch=1, train_flag=True, target_dir=''):
    train_loss_lst, train_accuracy_lst = [], []
    lenet_mode = lenet_mode.lower()
    lr = 0.1

    if not os.path.exists(target_dir + '/Saved_Train_Dict'):
        print('Creating ' + target_dir + '/Saved_Train_Dict directory')
        os.mkdir(target_dir + '/Saved_Train_Dict')

    if lenet_mode == 'dropout':
        net = LeNet5DropOut()
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif lenet_mode == 'weightdecay':
        net = LeNet5()
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-2)
    elif lenet_mode == 'batchnorm':
        net = LeNet5BatchNorm()
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif lenet_mode == 'regular':
        net = LeNet5()
        optimizer = optim.SGD(net.parameters(), lr=lr)
    else:
        raise IOError('Bad LeNet mode, select one of the following modes: DropOut / WeightDecay / BatchNorm / Regular')

    if train_flag:
        for epoch in range(n_epoch):
            start_time = time.time()
            net = net.to(device)
            if epoch == 0:
                net, epoch_loss, epoch_accuracy = single_epoch_train(net, optimizer, lenet_mode, active_train=False)
            else:
                net, epoch_loss, epoch_accuracy = single_epoch_train(net, optimizer, lenet_mode)
            print('Train - Epoch %d, Run time: %.2f [sec], Loss: %f, accuracy: %.2f%%' % (
            epoch, (time.time() - start_time), epoch_loss, 100 * epoch_accuracy))

            # Save net parameters after epoch done
            torch.save(net.state_dict(),
                       target_dir + '/Saved_Train_Dict/lenet5_' + lenet_mode + '_epoch_' + str(epoch) + '.pt')

            train_loss_lst.append(epoch_loss)
            train_accuracy_lst.append(epoch_accuracy)

    return net, train_loss_lst, train_accuracy_lst


def test(net, path_to_net=None):
    total_correct = 0.0

    if path_to_net is not None:
        # load net if there is path to saved model
        net.load_state_dict(torch.load(path_to_net))

    criterion = nn.CrossEntropyLoss()
    net.eval()

    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        loss = criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    accuracy = total_correct / len(data_test)

    return loss.detach().cpu().item(), accuracy.item()


def train_and_test_lenet5(lenet_mode='regular', n_epoch=1, run_mode='train', target_dir=''):
    global data_train_loader
    test_loss_lst, test_accuracy_lst = [], []
    test_loss, test_accuracy = -1, -1
    lenet_mode = lenet_mode.lower()
    run_mode = run_mode.lower()
    n_epoch = n_epoch + 1

    if 'train' in run_mode and 'test' in run_mode:
        train_flag, test_flag = True, True
        print('-- Starting train and test of ' + lenet_mode + ' LeNet5 --')
    elif 'train' in run_mode:
        train_flag, test_flag = True, False
        print('-- Starting train of ' + lenet_mode + ' LeNet5 --')
    elif 'test' in run_mode:
        train_flag, test_flag = False, True
        print('-- Starting test of ' + lenet_mode + ' LeNet5 --')
    else:
        raise IOError('Bad run mode selected, validate you have either test or train in run mode argument')

    net, train_loss_lst, train_accuracy_lst = train_lenet(lenet_mode, n_epoch, train_flag=train_flag,
                                                          target_dir=target_dir)

    if test_flag:
        net = net.to('cpu')
        for epoch in range(n_epoch):
            start_time = time.time()
            test_loss, test_accuracy = test(net,
                                            target_dir + '/Saved_Train_Dict/lenet5_' + lenet_mode + '_epoch_' + str(
                                                epoch) + '.pt')
            print('Test - Epoch %d, Run time: %.2f [sec], Loss: %f, accuracy: %.2f%%' % (
            epoch, (time.time() - start_time), test_loss, 100 * test_accuracy))

            test_loss_lst.append(test_loss)
            test_accuracy_lst.append(test_accuracy)

    now = datetime.now()

    loss_fig = plt.figure()
    loss_ax = loss_fig.add_subplot()
    if train_flag:
        loss_ax.plot(range(n_epoch), train_loss_lst, label='Train')
    if test_flag:
        loss_ax.plot(range(n_epoch), test_loss_lst, label='Test')
    loss_ax.legend(loc='upper right')
    loss_ax.set_title('Log Loss Vs #Epoch using ' + lenet_mode + ' LeNet5')
    if not os.path.exists(target_dir + '/Figures'):
        os.mkdir(target_dir + '/Figures')
    loss_fig.savefig(target_dir + '/Figures/Log_Loss_' + lenet_mode + '_mode_' + now.strftime("%H_%M") + '.png')

    acc_fig = plt.figure()
    acc_ax = acc_fig.add_subplot()
    if train_flag:
        acc_ax.plot(range(n_epoch), train_accuracy_lst, label='Train')
    if test_flag:
        acc_ax.plot(range(n_epoch), test_accuracy_lst, label='Test')
    acc_ax.legend(loc='upper left')
    acc_ax.set_title('Accuracy Vs #Epoch using ' + lenet_mode + ' LeNet5')
    acc_fig.savefig(target_dir + '/Figures/Accuracy_' + lenet_mode + '_mode_' + now.strftime("%H_%M") + '.png')
    plt.show(block=True)


if __name__ == '__main__':
    print('using ' + device + ' to train')
    if len(sys.argv) > 4:
        target_dir = sys.argv[4]
    else:
        target_dir = ''
    n_epoch = int(sys.argv[3])
    run_mode = sys.argv[2]
    lenet_mode = sys.argv[1]

    train_and_test_lenet5(lenet_mode, n_epoch, run_mode, target_dir)