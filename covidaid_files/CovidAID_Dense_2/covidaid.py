"""
The main CovidAID and CheXNet implementation
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features

        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(1000, 100)
        self.fc2 = torch.nn.Linear(100, 250)
        self.fc3 = torch.nn.Linear(250, 500)
        self.fc4 = torch.nn.Linear(500, 250)
        self.fc5 = torch.nn.Linear(250, 100)
        self.fc6 = torch.nn.Linear(100, out_size)
        self.drop = torch.nn.Dropout(0.5)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc6(x)
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
