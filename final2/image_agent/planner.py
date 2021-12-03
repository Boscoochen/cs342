import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torchvision import transforms
def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)



class Planner(torch.nn.Module):
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
        self.conv1 = Conv2d(3,16,3,1,1)
        self.bn1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(16,16,3,1,1)
        self.bn2 = BatchNorm2d(16)
        self.relu2 = ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv3 = Conv2d(16,32,3,1,1)
        self.bn3 = BatchNorm2d(32)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(32,32,3,1,1)
        self.bn4 = BatchNorm2d(32)
        self.relu4 = ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv5 = Conv2d(32,64,3,1,1)
        self.bn5 = BatchNorm2d(64)
        self.relu5 = ReLU()
        self.conv6 = Conv2d(64,64,3,1,1)
        self.bn6 = BatchNorm2d(64)
        self.relu6 = ReLU()
        self.conv7 = Conv2d(64,64,3,1,1)
        self.bn7 = BatchNorm2d(64)
        self.relu7 = ReLU()
        self.maxpool3 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        


        self.conv8 = Conv2d(64,128,3,1,1)
        self.bn8 = BatchNorm2d(128)
        self.relu8 = ReLU()
        self.conv9 = Conv2d(128,128,3,1,1)
        self.bn9 = BatchNorm2d(128)
        self.relu9 = ReLU()
        self.conv10 = Conv2d(128,128,3,1,1)
        self.bn10 = BatchNorm2d(128)
        self.relu10 = ReLU()
        self.maxpool4 = nn.MaxPool2d(2,2,padding=0, dilation=1)
        

        self.conv11 = Conv2d(128,128,3,1,1)
        self.bn11 = BatchNorm2d(128)
        self.relu11 = ReLU()
        self.conv12 = Conv2d(128,128,3,1,1)
        self.bn12 = BatchNorm2d(128)
        self.relu12 = ReLU()
        self.conv13 = Conv2d(128,128,3,1,1)
        self.bn13 = BatchNorm2d(128)
        self.relu13 = ReLU()
        self.maxpool5 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv_trans1 = nn.Conv2d(128, 64, 1)
        self.conv_trans2 = nn.Conv2d(64, 1, 1)
        self.upsample_2x = nn.ConvTranspose2d(128, 128, 4, 2, 1,output_padding=(1,0), bias=False)
        self.upsample_2x_no_outpadding = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False)
        self.upsample_2x_1 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.upsample_2x_2 = nn.ConvTranspose2d(1, 1, 16, 8, 4, bias=False)
        self.upsample_2x_2_1 = nn.ConvTranspose2d(1, 1, 8, 4, 2, bias=False)
        self.upsample_2x_2_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsample_8x = nn.ConvTranspose2d(1, 1, 16, 8, 4, bias=False)

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
        # print(x.shape)
        #1,3,16,16
        transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x = transform(x)
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.relu2(x5)
        if x6.shape[2] != 1 and x6.shape[3] != 1:
          count += 1
          x6 = self.maxpool1(x6)


        x7 = self.conv3(x6)
        x8 = self.bn3(x7)
        x9 = self.relu3(x8)
        x10 = self.conv4(x9)
        x11 = self.bn4(x10)
        x12 = self.relu4(x11)
        if x12.shape[2] != 1 and x12.shape[3] != 1:
          count += 1
          x12 = self.maxpool2(x12)
        #1,64,2,2
        x13 = self.conv5(x12)
        x14 = self.bn5(x13)
        x15 = self.relu5(x14)
        x16 = self.conv6(x15)
        x17 = self.bn6(x16)
        x18 = self.relu6(x17)
        x19 = self.conv7(x18)
        x20 = self.bn7(x19)
        x21 = self.relu7(x20)
        if x21.shape[2] != 1 and x21.shape[3] != 1:
          count += 1
          x21 = self.maxpool3(x21)
        # print(x15.shape)
        #1,128,1,1

        x22 = self.conv8(x21)
        x23 = self.bn8(x22)
        x24 = self.relu8(x23)
        x25 = self.conv9(x24)
        x26 = self.bn9(x25)
        x27 = self.relu9(x26)
        x28 = self.conv10(x27)
        x29 = self.bn10(x28)
        x30 = self.relu10(x29)
        if x30.shape[2] != 1 and x30.shape[3] != 1:
          count += 1
          x30 = self.maxpool4(x30)
        

        x31 = self.conv11(x30)
        x32 = self.bn11(x31)
        x33 = self.relu11(x32)
        x34 = self.conv12(x33)
        x35 = self.bn12(x34)
        x36 = self.relu12(x35)
        x37 = self.conv13(x36)
        x38 = self.bn13(x37)
        x39 = self.relu13(x38)
        if x39.shape[2] != 1 and x39.shape[3] != 1:
          count += 1
          x39 = self.maxpool5(x39)
        

        if count != 0:
            x39 = self.upsample_2x_no_outpadding(x39)
            count -= 1
            
            
        if x39.shape == x30.shape:
            x39 = x39 + x30
        x39 = self.conv_trans1(x39)
        # print(add1.shape)
        if count!= 0:
            x39 = self.upsample_2x_1(x39)
            count -= 1
        # print(add1.shape)
        if x39.shape == x21.shape:
            x39 = x39 + x21
        x39 = self.conv_trans2(x39)
        # print(x33.shape)
        if count != 0:
          if count == 3:
              x39 = self.upsample_2x_2(x39)
              count -= 1
          elif count == 2:
              x39 = self.upsample_2x_2_1(x39)
              count -= 1
          elif count == 1:
              x39 = self.upsample_2x_2_2(x39)
        # z = z.permute(0,2,3,1)[:,:,:,-1]
        # print(z.shape)
        return (1 + spatial_argmax(x39.squeeze(1))) * torch.as_tensor([x.size(3) - 1, x.size(2) - 1]).float().to(x.device)




def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
