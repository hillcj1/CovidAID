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

        self.conv1 = torch.nn.Conv1d(1000, 256, 1, 1)
        self.relu = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(256)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, out_size)
        self.dropout = torch.nn.Dropout(0.5)

        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)

        x = x[:, :, np.newaxis]
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.batch1(x)

        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
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
