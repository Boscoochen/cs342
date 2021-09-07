import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import ReLU
from torch import nn
from torch.nn import BatchNorm2d

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        self.input = input
        #torch.Size([4,3])
        self.target = target
        #torch.Size([4])
        # print(self.target)
        self.loss = nn.CrossEntropyLoss()
        self.output = self.loss(self.input, self.target)
        return self.output
        raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.linear1 = Linear(3*64*64, 6)
        #raise NotImplementedError('LinearClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.linear1((x.view(x.size(0), -1)))
        raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        # print("here")
        self.BN = BatchNorm2d(3)
        self.linear1 = Linear(3*64*64, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 6)
        # raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
       
        x = self.BN(x)
        x = self.linear1((x.view(x.size(0), -1)))
        x = self.relu(x)
        x = self.linear2((x.view(x.size(0), -1)))

        return x
        raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
