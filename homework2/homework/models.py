import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import BatchNorm2d
class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super(CNNClassifier, self).__init__()
        self.conv1_bn = BatchNorm2d(3)
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)
        self.relu1 = ReLU()
        self.conv2_bn = BatchNorm2d(6)
        self.linear1 = Linear(6*62*62, 100)
        self.relu2 = ReLU()
        self.linear2 = Linear(100, 6)
        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # print(x.shape)
        x = self.conv1_bn(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2_bn(x)
        x = self.linear1((x.view(x.size(0), -1)))
        x = self.relu2(x)
        return self.linear2((x.view(x.size(0), -1)))
        # print(x.shape)
        # return x
        raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
