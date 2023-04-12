import torch
from torchsummary import summary
import os
import argparse
import time

from models.ResNet import ResNet50
from utils import DataLoader, Logger    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        help="specify which device to run the task",
        type=str,
        default=""
    )
    parser.add_argument(
        "--download",
        help="specify whether to use the local dataset or download the dataset",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--epoch",
        help="total training&eval epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--lr",
        help="learning rate",
        type=float,
        default=0.01
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    if args.device == "cpu":
        DEVICE = "cpu"
    elif args.device == "gpu" or args.device == "dcu" or args.device == "cuda":
        DEVICE = "cuda"
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    DOWNLOAD = args.download
    ROUND = args.epoch
    LEARNING_RATE = args.lr
    PATH_TO_DATA = './data/'
    LOG_NAME = "./logs/ResNetCifar10_{}.log".format(DEVICE)

    MyLogger = Logger.Logger(log_name=LOG_NAME, append=False)
    logger = MyLogger.logger
    progress_bar = MyLogger.progress_bar

    model = ResNet50().to(device=DEVICE)
    _ = summary(model, (3,32,32), device=DEVICE)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader, classes = DataLoader.load_data_cifar10(
        path_to_data=PATH_TO_DATA, download=DOWNLOAD
    )
    logger.info("train_loader: len={} batch_size={}".format(train_loader.__len__(), train_loader.batch_size))
    logger.info("test_loader: len={} batch_size={}".format(test_loader.__len__(), test_loader.batch_size))
    logger.info("classes: labels={}".format(classes))
    best_acc, best_epoch = 0, 0

    def train(epoch):
        logger.info("training epoch {} start".format(epoch))
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    def evalaution(epoch):
        global best_acc, best_epoch
        logger.info("evaluation epoch {} start".format(epoch))
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            logger.info('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
            best_epoch = epoch

        logger.info("current best acc={} in epoch={}".format(best_acc, best_epoch))

    start_time = time.time()
    for epoch in range(ROUND):
        train(epoch=epoch)
        evalaution(epoch=epoch)
    end_time = time.time()
    total_seconds = end_time-start_time
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    logger.info("total consuming time: %d hours %d minutes %s seconds"%(hours, minutes, seconds))