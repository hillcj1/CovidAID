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

        self.conv1 = torch.nn.Conv1d(num_ftrs, 64, 1, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.batch = torch.nn.BatchNorm1d(64)

        self.fc1 = torch.nn.Linear(14400, 128)
        self.fc2 = torch.nn.Linear(128, out_size)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        print("Initial shape: " + str(x.shape))
        x = self.densenet121(x)
        print("After DenseNet: " + str(x.shape))
        x = x[:, :, np.newaxis]
        print("After Reshape: " + str(x.shape))
        x = self.conv1(x)
        print("After Conv1: " + str(x.shape))
        x = self.dropout(x)
        print("After Dropout: " + str(x.shape))
        x = self.relu(x)
        print("After ReLU: " + str(x.shape))
        #x = self.pool(x)
        x = self.batch(x)
        print("After Batch: " + str(x.shape))
        b = x.shape[0]
        x = torch.reshape(x, (b, int(28800 / b)))
        print("After Reshape2: " + str(x.shape))
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
