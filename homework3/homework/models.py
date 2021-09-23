import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import BatchNorm2d

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        super(CNNClassifier, self).__init__()
        self.conv1 = Conv2d(3,64,7,2,3)
        self.conv1_bn = BatchNorm2d(64)
        self.relu1 = ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2_bn = BatchNorm2d(64)
        self.linear1 = Linear(64*16*16, 150)
        self.relu2 = ReLU()
        self.linear2 = Linear(150, 6)
        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2_bn(x)
        x = self.linear1((x.view(x.size(0), -1)))
        x = self.relu2(x)
        return self.linear2((x.view(x.size(0), -1)))
        raise NotImplementedError('CNNClassifier.forward')



class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.conv1 = Conv2d(3,6,3,1,1)
        self.bn1 = BatchNorm2d(6)
        self.relu1 = ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2,padding=0, dilation=1)




        self.conv2 = Conv2d(6,32,3,1,1)
        self.bn2 = BatchNorm2d(32)
        self.relu2 = ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv3 = Conv2d(32,64,3,1,1)
        self.bn3 = BatchNorm2d(64)
        self.relu3 = ReLU()
        self.maxpool3 = nn.MaxPool2d(2,2,padding=0, dilation=1)
        self.upsample_2x_1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.upsample_2x_2 = nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False)
        self.upsample_2x_3 = nn.ConvTranspose2d(5, 5, 4, 2, 1, bias=False)
        self.conv_trans1 = nn.Conv2d(64, 32, 1)
        self.conv_trans2 = nn.Conv2d(32, 5, 1)

        self.conv4 = Conv2d(64,64,3,1,1)
        self.bn4 = BatchNorm2d(64)
        self.relu4 = ReLU()
        self.maxpool4 = nn.MaxPool2d(2,2,padding=0, dilation=1)
        # self.conv2 = Conv2d(64,128,3,1,1)
        # raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        count = 0
        check = True
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        if x3.shape[2] != 1 and x3.shape[3] != 1:
          count += 1
          x3 = self.maxpool1(x3)
        x5 = self.conv2(x3)
        x6 = self.bn2(x5)
        x7 = self.relu2(x6)
        if x7.shape[2] != 1 and x7.shape[3] != 1:
          count += 1
          x7 = self.maxpool2(x7)
        x9 = self.conv3(x7)
        x10 = self.bn3(x9)
        x11 = self.relu3(x10)
        if x11.shape[2] != 1 and x11.shape[3] != 1:
          count += 1
          x11 = self.maxpool3(x11)
        x13 = self.conv4(x11)
        x14 = self.bn4(x13)
        x15 = self.relu4(x14)
        if x15.shape[2] != 1 and x15.shape[3] != 1:
          count += 1
          x15 = self.maxpool4(x15)
        if count != 0:
          x15 = self.upsample_2x_1(x15)
          count -= 1
        if x15.shape == x11.shape:
          # print("in1")
          x15 = x15 + x11
        x15 = self.conv_trans1(x15)
        if count != 0:
          x15 = self.upsample_2x_2(x15)
          count -= 1
        if x15.shape == x7.shape:
          # print("in2")
          x15 = x15 + x7
        x15 = self.conv_trans2(x15)
        if count != 0:
          x15 = self.upsample_2x_3(x15)
          count -= 1
        if count != 0:
          x15 = self.upsample_2x_3(x15)
          count -= 1

        return x15

        raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
