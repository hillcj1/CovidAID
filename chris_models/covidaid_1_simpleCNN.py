"""
The main CovidAID and CheXNet implementation
"""

import torch
import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features

        self.conv1 = torch.nn.Conv2d(num_ftrs, 64, 4, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(num_ftrs, 2)
        self.batch = torch.nn.BatchNorm2d(64)

        self.fc1 = torch.nn.Linear(14400, 128)
        self.fc2 = torch.nn.Linear(128, out_size)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batch(x)
        b = x.shape[0]
        x = torch.reshape(x, (b, int(28800 / b)))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x
    

class CovidAID(DenseNet121):
    """
    Modified DenseNet network with 4 classes
    """
    def __init__(self, combine_pneumonia=False):
        NUM_CLASSES = 3 if combine_pneumonia else 4
        super(CovidAID, self).__init__(NUM_CLASSES)


class CheXNet(DenseNet121):
    """
    Modified DenseNet network with 14 classes
    """
    def __init__(self):
        super(CheXNet, self).__init__(14)
