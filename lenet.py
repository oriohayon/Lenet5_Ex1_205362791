import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    LeNet5 famous CNN proposed by LeCun et al. (1998).
    """
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet5, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(data_in))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # classification
        x = x.view(data_in.size(0), -1)# flatten
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


class LeNet5DropOut(nn.Module):
    """
    LeNet5 using a dropout
    """
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet5, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(data_in))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # classification
        x = x.view(data_in.size(0), -1)# flatten
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x

class LeNet5BatchNorm(nn.Module):
    """
    LeNet5 using a dropout
    """
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet5, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(1024, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(data_in))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # classification
        x = x.view(data_in.size(0), -1)# flatten
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.softmax(self.fc2_bn(self.fc2(x)))

        return x


class LeNet5WeightDecay(nn.Module):
    """
    LeNet5 using a dropout
    """
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet5, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(data_in))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # classification
        x = x.view(data_in.size(0), -1)# flatten
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x