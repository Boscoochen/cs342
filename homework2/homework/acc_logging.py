from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    step = 0
    train_acc = []
    valid_acc = []
    for epoch in range(10):
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_acc.append(sum(dummy_train_accuracy)/len(dummy_train_accuracy))
            train_logger.add_scalar('loss', dummy_train_loss, step)
            step = step + 1
            # raise NotImplementedError('Log the training loss')
        # print(len(train_acc))
        train_logger.add_scalar('accuracy', sum(train_acc)/len(train_acc),step)
        train_acc = []
        # raise NotImplementedError('Log the training accuracy')
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            valid_acc.append(sum(dummy_validation_accuracy)/len(dummy_validation_accuracy))
        valid_logger.add_scalar('accuracy', sum(valid_acc)/len(valid_acc), step)
        valid_acc = []
        # raise NotImplementedError('Log the validation accuracy')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
