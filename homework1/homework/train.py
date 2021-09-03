from .models import ClassificationLoss, model_factory, save_model,load_model
from .utils import accuracy, load_data, SuperTuxDataset
import torch

def train(args):
    model = model_factory[args.model]()
   
    """
    Your code here

    """
    print("code here")
    traindata_path = "data/train"
    testdata_path = "data/valid"

    train_dataloader = load_data(traindata_path, num_workers=0, batch_size=128)
    test_dataloader = load_data(testdata_path, num_workers=0, batch_size=128)

    print(model)
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_train_step = 0
    total_test_step = 0
    epoch = 10

    for i in range(epoch):
      for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = ClassificationLoss().forward(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
          print("times train: {}, Loss: {}".format(total_train_step,loss))
    
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
      for data in test_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = ClassificationLoss().forward(outputs, targets)
        total_test_loss = total_test_loss + loss
        acc = accuracy(outputs,targets) 
        total_accuracy = total_accuracy = acc
    print("total test loss: {}".format(total_test_loss))
    return total_accuracy
    #raise NotImplementedError('train')
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
