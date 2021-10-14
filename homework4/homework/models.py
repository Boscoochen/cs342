import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torchvision import transforms
from . import dense_transforms
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    # print(type(heatmap))
    #150 * 100,  10*10,  54*123, 100*100
    count = 0
    temp_map = heatmap
    # print("heatmap size", heatmap.shape)
    # v1, indices1 = torch.topk(temp_map.flatten(), k=k,largest=True)

    # indices1 = torch.tensor(np.array(np.unravel_index(indices1.numpy(), temp_map.shape)).T)

    # print(temp_map.shape)
    # print(max_det)
    # print(max_pool_ks)
    # print(min_score)
    k=0
    if max_det >= heatmap.shape[0]*heatmap.shape[1]:
      k = heatmap.shape[0]*heatmap.shape[1]
    else:
      k = max_det
    # print("k", k)
    pad = (max_pool_ks-1)//2
    heatmap = heatmap[None, None, :]
    hmax = F.max_pool2d(heatmap, (max_pool_ks,max_pool_ks),stride=1,padding=pad)
    keep = (hmax == heatmap)
    # print("hmax size", hmax.shape)
    # print("heatmap size", heatmap.shape)
    keep = heatmap*keep
    # print(keep)
    
    batch, cat, height, width = keep.size()
    topk_scores, topk_inds = torch.topk(keep.view(batch, cat, -1), k)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
    topk_clses = (topk_ind / k).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

    # print("topk_score", topk_score.shape)
    # print("topk_ys", topk_ys.shape)
    # print("topk_xs", topk_xs.shape)
    # keep = keep.reshape(keep.shape[2],keep.shape[3])

    # v, indices = torch.topk(keep.flatten(), k=k,largest=True)
    # indices = torch.tensor(np.array(np.unravel_index(indices.numpy(), keep.shape)).T)
   

    small_container = []
    large_container = []
    # print(len(v))
    # print(len(topk_score[0]))
    for i in range(len(topk_score[0])):
      if topk_score[0][i] > min_score:
        small_container.append(topk_score[0][i].float())
        small_container.append(topk_xs[0][i].int())
        small_container.append(topk_ys[0][i].int())
        large_container.append(small_container)
        small_container=[]

   
    # print(large_container)
    return large_container
 
    raise NotImplementedError('extract_peak')

class Detector(torch.nn.Module):
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        self.conv1 = Conv2d(3,64,3,1,1)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(64,64,3,1,1)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv3 = Conv2d(64,128,3,1,1)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(128,128,3,1,1)
        self.bn4 = BatchNorm2d(128)
        self.relu4 = ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv5 = Conv2d(128,256,3,1,1)
        self.bn5 = BatchNorm2d(256)
        self.relu5 = ReLU()
        self.conv6 = Conv2d(256,256,3,1,1)
        self.bn6 = BatchNorm2d(256)
        self.relu6 = ReLU()
        self.conv7 = Conv2d(256,256,3,1,1)
        self.bn7 = BatchNorm2d(256)
        self.relu7 = ReLU()
        self.maxpool3 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        

        self.conv8 = Conv2d(256,512,3,1,1)
        self.bn8 = BatchNorm2d(512)
        self.relu8 = ReLU()
        self.conv9 = Conv2d(512,512,3,1,1)
        self.bn9 = BatchNorm2d(512)
        self.relu9 = ReLU()
        self.conv10 = Conv2d(512,512,3,1,1)
        self.bn10 = BatchNorm2d(512)
        self.relu10 = ReLU()
        self.maxpool4 = nn.MaxPool2d(2,2,padding=0, dilation=1)
        

        self.conv11 = Conv2d(512,512,3,1,1)
        self.bn11 = BatchNorm2d(512)
        self.relu11 = ReLU()
        self.conv12 = Conv2d(512,512,3,1,1)
        self.bn12 = BatchNorm2d(512)
        self.relu12 = ReLU()
        self.conv13 = Conv2d(512,512,3,1,1)
        self.bn13 = BatchNorm2d(512)
        self.relu13 = ReLU()
        self.maxpool5 = nn.MaxPool2d(2,2,padding=0, dilation=1)

        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, 3, 1)
        self.upsample_2x = nn.ConvTranspose2d(512, 512, 4, 2, 1,output_padding=(1,0), bias=False)
        self.upsample_2x_no_outpadding = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.upsample_2x_1 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_2x_2 = nn.ConvTranspose2d(3, 3, 16, 8, 4, bias=False)
        self.upsample_2x_2_1 = nn.ConvTranspose2d(3, 3, 8, 4, 2, bias=False)
        self.upsample_2x_2_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsample_8x = nn.ConvTranspose2d(3, 3, 16, 8, 4, bias=False)
        

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        count = 0
        check = True
        #1,3,16,16
        # print(x.shape)
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
        return x39
        raise NotImplementedError('Detector.forward')

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        
        # image = image.permute(1,2,0)[:,:,-1]
        for i in range(image.shape[0]):
          list_of_peaks= extract_peak(image[0])

        

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))

def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r

if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
