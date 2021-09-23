import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from torch import nn


def train(args):
    from os import path
    device1 = torch.device("cuda")
    device2 = torch.device("cpu")
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    print("code here")
        
    traindata_path = "dense_data/train"
    testdata_path = "dense_data/valid"

    loss_fun = nn.CrossEntropyLoss().to(device1)
    train_dataloader = load_dense_data(traindata_path, num_workers=4, batch_size=32)
    test_dataloader = load_dense_data(testdata_path, num_workers=4, batch_size=32)

    # print(train_dataloader)
    # print(test_dataloader)


    # print(model)
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_train_step = 0
    total_test_step = 0
    epoch = 10

    for i in range(epoch):
      model.train()
      for data in train_dataloader:
        imgs, targets = data
        # print(type(imgs))
        # print(len(targets))
        # print(imgs.shape)
        #torch.Size([128, 3, 64, 64])
        outputs = model(imgs).to(device1)
        #torch.Size([128, 6, 62, 62])
        # print(outputs.shape)
        # outputs = outputs.view(outputs.size(0), -1)
        # print(outputs.shape)
        targets =targets.long().to(device2)
        
        
        targets = targets.to(device1)
        loss = loss_fun(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        # print(total_train_step)
        if total_train_step % 100 == 0:
          print("times train: {}, Loss: {}".format(total_train_step,loss))
    
    model.eval()
    total_test_loss = 0
    acc = []
    confusion = ConfusionMatrix(6)
    with torch.no_grad():
      for data in test_dataloader:
        imgs, targets = data
        outputs = model(imgs).to(device1)
        # outputs = outputs.view(outputs.size(0), -1)
        targets = targets.long().to(device2)
        targets = targets.to(device1)
        loss = loss_fun(outputs, targets)
        total_test_loss = total_test_loss + loss
        confusion.add(outputs.argmax(1), targets)
        outputs_idx = outputs.max(1)[1].type_as(targets)
        acc.append(outputs_idx.eq(targets).float().mean())
        
    print("total test loss: {}".format(total_test_loss))
    print(confusion.global_accuracy)
    
    #raise NotImplementedError('train')
    save_model(model)
    print(sum(acc)/len(acc))
    return sum(acc)/len(acc)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-m', '--model', default='fcn')
    args = parser.parse_args()
    train(args)
