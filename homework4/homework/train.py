import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb

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
    # loss_fun = nn.CrossEntropyLoss()
    train_dataloader = load_detection_data(traindata_path)
    test_dataloader = load_detection_data(testdata_path)

    model_dict = model.state_dict()
    model.load_state_dict(model_dict)

    print(type(train_dataloader.dataset))
    # train_np = torch.tensor(np.array(train_dataloader.dataset[1:4]))
    # print(train_dataloader)
    # print(test_dataloader)

    # print(model)
    # learning_rate = 0.001
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_train_step = 0
    total_test_step = 0
    epoch = 1
    count = 0
    for i in range(epoch):
      model.train()
      for data in train_dataloader.dataset:
        im, karts, bombs, pickup = data
        # print(type(im))
        # print(type(karts))
        # print(type(bombs))
        # print(type(pickup))
        # break
        im, karts, bombs, pickup = im.to(device),torch.from_numpy(karts.astype(float)).to(device),torch.from_numpy(bombs.astype(float)).to(device),torch.from_numpy(pickup.astype(float)).to(device)
        
        # print("im size",im.shape)
        # print("karts size",karts.shape)
        # print("bombs size",bombs.shape)
        # print("pickup size",pickup.shape)
        # count+=1
        # print("//")
        # if count == 10:
        #   break
    #     transform = dense_transforms.RandomHorizontalFlip()
    #     transform2 = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
    #     imgs = transform2(imgs)
    #     imgs, targets = transform(imgs, targets)
    #     imgs, targets = imgs.to(device), targets.long().to(device)
        # outputs = model(a[None,:])
        det = torch.cat((karts, bombs, pickup), 0)
        det = det.cpu().numpy()
        t = dense_transforms.ToHeatmap()
        image, heatmap, size = t(im,det)
        print(heatmap)
        image = model(im[None,:])
 
        
        image = image.permute(0,1,2,3)[-1,:,:,:]
        det_heatmap = model.detect(image)
        # t = dense_transforms.ToHeatmap()
        # image, heatmap, size = t(image,det)
        # print(heatmap)

    #     loss = loss_fun(outputs, targets)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     total_train_step = total_train_step + 1
    #     if total_train_step % 100 == 0:
    #       print("times train: {}, Loss: {}".format(total_train_step,loss))
    
    # model.eval()
    # total_test_loss = 0
    # with torch.no_grad():
    #   for data in test_dataloader:
    #     imgs, targets = data
    #     imgs, targets = imgs.to(device), targets.long().to(device)
    #     outputs = model(imgs)
    #     loss = loss_fun(outputs, targets)
    #     total_test_loss = total_test_loss + loss
        
        
    # print("total test loss: {}".format(total_test_loss))
    

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
