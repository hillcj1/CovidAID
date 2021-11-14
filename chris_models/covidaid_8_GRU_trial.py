"""
The main CovidAID and CheXNet implementation
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable
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

        self.batch_size = 1
        self.hidden_dim = 1000
        self.n_layers = 2

        self.gru = torch.nn.GRU(1000, hidden_dim, n_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, out_size)
        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.densenet121(x)
        x = x[:, :, np.newaxis]

        hidden = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
        output, hidden = self.gru(x, hidden) 
        out = output.contiguous().view(-1, self.hidden_dim) 
        out = self.relu(out)

        out = out[:, :, 0]

        out = self.fc1(out)
        out = self.sig(out)
        return out
    

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
