from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.nn as nn

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    print("code here")
    traindata_path = "data/train"
    testdata_path = "data/valid"

    loss_fun = nn.CrossEntropyLoss()

    train_dataloader = load_data(traindata_path, num_workers=0, batch_size=128)
    test_dataloader = load_data(testdata_path, num_workers=0, batch_size=128)

    print(train_dataloader)
    print(test_dataloader)


    print(model)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_train_step = 0
    total_test_step = 0
    epoch = 15

    for i in range(epoch):
      model.train()
      for data in train_dataloader:
        imgs, targets = data
        # print(len(targets))
        # print(imgs.shape)
        #torch.Size([128, 3, 64, 64])
        outputs = model(imgs)
        #torch.Size([128, 6, 62, 62])
        # print(outputs.shape)
        # outputs = outputs.view(outputs.size(0), -1)
        # print(outputs.shape)
        loss = loss_fun(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
          print("times train: {}, Loss: {}".format(total_train_step,loss))
    
    model.eval()
    total_test_loss = 0
    acc = []
    confusion = ConfusionMatrix(6)
    with torch.no_grad():
      for data in test_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        # outputs = outputs.view(outputs.size(0), -1)
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-m', '--model', default='cnn')
    args = parser.parse_args()
    train(args)
