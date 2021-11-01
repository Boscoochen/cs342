import torch
import numpy as np
import torch.nn.functional as F
from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
from torch import nn


def train(args):
    from os import path
    device = torch.device('cuda')
    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    traindata_path = "dense_data/train"
    testdata_path = "dense_data/valid"
    dense = [0.02929112, 0.0044619, 0.00411153]
    # w = torch.as_tensor(dense)
    dense_sum = sum(dense)
    for i in range(3):
      dense[i] = (1 - dense[i]) / dense[i]
    w1 = torch.full((96,128), 0.02929112)
    w1 = w1.unsqueeze(dim=2)
    w2 = torch.full((96,128), 0.0044619)
    w2 = w2.unsqueeze(dim=2)
    w3 = torch.full((96,128), 0.00411153)
    w3 = w3.unsqueeze(dim=2)
    w = torch.cat([w1, w2, w3], dim=2)
    w = w.permute(2,0,1)
    # print(sum(dense))
    print(dense)

    pred_class = torch.rand (3,96,128)
    for j in range(3):
      pred_class[j] = dense[j]
    
    print(pred_class.shape)
    print(pred_class)
    



    BCELogLoss =torch.nn.BCEWithLogitsLoss(pos_weight=pred_class).to(device)
    # loss_fun = nn.CrossEntropyLoss()
    transform = dense_transforms.Compose([
      dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
      dense_transforms.RandomHorizontalFlip(),
      dense_transforms.ToTensor(),
      dense_transforms.ToHeatmap()
    ])

    train_dataloader = load_detection_data(traindata_path,num_workers=4,transform=transform)


    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

    total_train_step = 0
    total_test_step = 0
    epoch = 20

    for i in range(epoch):
      model.train()
      for data in train_dataloader:
        image, label, size = data
        image, label, size = image.to(device),label.to(device),size.to(device)
        outputs = model(image)
        # b,c,h,w = outputs.shape
        # for batch in range(b):
        #   c1,c2,c3 = model.detect(outputs[batch])
        m = nn.Sigmoid()
        # loss = BCELoss(m(outputs), label)
        # outputs = outputs.permute(0,2,3,1)
        # label = label.permute(0,2,3,1)
        # label = label.permute(1,2,3,0)[:,:,:,-1]
        # print(label.shape)
        loss = BCELogLoss(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
          print("times train: {}, Loss: {}".format(total_train_step,loss))
    
    save_model(model)

def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-m', '--model', default='Detector')
    args = parser.parse_args()
    train(args)
