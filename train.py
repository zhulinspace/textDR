import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms
import itertools
import os
import time
import argparse

from dataset import PlayDataset
from model import Model
from config import opt

import warnings
warnings.filterwarnings("ignore")

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.Resize((32,640)),
    transforms.ToTensor()
])

def train(op):
    net = Model(op)
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    train_dataset = PlayDataset(is_train=True, train_val=0.9, transform=transform_test)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    testdataset = PlayDataset(is_train=False, train_val=0.9, transform=transform_test)
    testloader = DataLoader(testdataset, batch_size=16, shuffle=True, num_workers=8)

    if 'CTC' in op.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion=torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    '''加载部分权重模型'''

    '''多gpu部分'''

    for epoch in range(10000):
        print('\nEpoch: %d' % epoch)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, sample_batch in enumerate(trainloader):
            inputs = sample_batch['image'].type(torch.FloatTensor).to(device)
            targets = sample_batch['label'].type(torch.IntTensor).to(device)
            preds = net(inputs, None).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * targets.shape[0])

            # '''计算正确率'''
            # y_pred_labels = []
            # for sample in preds:
            #     _, y = sample.max(1)
            #     y = [int(k) for k, g in itertools.groupby(y)]  # 去掉重复的
            #     while 0 in y:  # 去掉0
            #         y.remove(0)
            #     y_pred_labels.append(y)
            # for pred, label in zip(y_pred_labels, targets.tolist()):
            #     if (pred == label):
            #         correct += 1

            preds = preds.permute(1, 0, 2)
            length = sample_batch['real_length']
            cost = criterion(preds, targets, preds_size.to(device), length.to(device))
            net.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)  # gradient clipping with 5 (Default)
            optimizer.step()

            train_loss += cost.item()
            total += targets.size(0)


        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(testloader):
                inputs = sample_batch['image'].type(torch.FloatTensor).to(device)
                targets = sample_batch['label'].type(torch.IntTensor).to(device)
                preds = net(inputs, None).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * targets.shape[0])
                '''查看输出的大小值'''
                # print('inputs.shape',inputs.shape())
                # print('\n targets.shape',targets.shape())
                # '''计算正确率 写的计算正确率是不是有问题 肯定是有问题'''
                # y_pred_labels = []
                # for sample in preds:
                #     _, y = sample.max(1)
                #     y = [int(k) for k, g in itertools.groupby(y)]  # 去掉重复的
                #     while 0 in y:  # 去掉0
                #         y.remove(0)
                #     y_pred_labels.append(y)
                # for pred, label in zip(y_pred_labels, targets.tolist()):
                #     if (pred == label):
                #         correct += 1
                preds = preds.permute(1, 0, 2)
                length = sample_batch['real_length']
                cost = criterion(preds, targets, preds_size.to(device), length.to(device))
                test_loss += cost.item()
                total += targets.size(0)
            sample = preds.permute(1, 0, 2)[0]
            _, predicted = sample.max(1)
            print(predicted)
            print(targets[0])
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # save model
    state = {
            'net': net.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/first.t7')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="None", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    parser.add_argument('--num_iter',type=int,default=10000,help='number of iterations to train for')
    op = parser.parse_args()
    op.num_class = 1557
    train(op)
















