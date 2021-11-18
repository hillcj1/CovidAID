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

        self.conv1 = torch.nn.Conv1d(num_ftrs, 128, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(128, 128, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv1d(128, 128, kernel_size=1, stride=1)
        self.conv4 = torch.nn.Conv1d(128, 256, kernel_size=1, stride=1)
        self.conv5 = torch.nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.conv6 = torch.nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.conv7 = torch.nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.conv8 = torch.nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.conv9 = torch.nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.conv10 = torch.nn.Conv1d(256, 512, kernel_size=1, stride=1)
        self.conv11 = torch.nn.Conv1d(512, 512, kernel_size=, stride=1)

        self.relu = torch.nn.ReLU()

        self.pool = torch.nn.MaxPool1d(3, 3)

        self.batch1 = torch.nn.BatchNorm1d(128)
        self.batch2 = torch.nn.BatchNorm1d(256)
        self.batch3 = torch.nn.BatchNorm1d(512)

        self.fc = torch.nn.Linear(512, out_size)
  
        self.dropout = torch.nn.Dropout(0.5)

        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv7(x)
        x = self.batch2(x)
        x = self.relu(x)
        
        x = self.conv8(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv9(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv10(x)
        x = self.batch3(x)
        x = self.relu(x)

        x = self.conv11(x)
        x = self.batch3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.fc1(x)
        #x = self.relu(x)
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