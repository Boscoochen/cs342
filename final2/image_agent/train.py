from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    device = torch.device('cuda')
    model = Planner().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    traindata_path = "test_csv_2"
    MSEloss = torch.nn.L1Loss(reduction='sum')
    transform = dense_transforms.Compose([
      dense_transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
      dense_transforms.RandomHorizontalFlip(),
      dense_transforms.ToTensor()
    ])

    train_dataloader = load_data(traindata_path,num_workers=4)


    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

    total_train_step = 0
    total_test_step = 0
    epoch = 150
    batch_reg_masks = []
    for i in range(epoch):
      model.train()
      for data in train_dataloader:
        image, label = data
        image, label= image.to(device),label.to(device)

        outputs= model(image)
        h, w = image.size()[2], image.size()[3]
        # print(outputs.shape)
        # print(label.shape)
        x,y = label.chunk(2, dim=1)
        xy = torch.cat((x.clamp(min=0.0,max=w),y.clamp(min=0.0,max=h)),dim=1)
        xy = xy.to(device)

        loss = MSEloss(outputs, xy)*0.01
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
          print("times train: {}, Loss: {}".format(total_train_step,loss))

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-m', '--model', default='planner')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
