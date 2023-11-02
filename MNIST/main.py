# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from spiking_model import*
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

names = 'snn_thresholding'
data_path =  'E:\Dataset\MNIST' #todo: input your data path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
threshold_history = list([])

snn = SCNN()
# lifneuron = LIFNeuron()
snn.to(device)

criterion = nn.MSELoss()

# optimizer = torch.optim.Adam([{'params': snn.parameters()}, {'params': snn.w}], lr=learning_rate)

optimizer = torch.optim.Adam([{'params': snn.parameters()}], lr=learning_rate)
# optimizer = torch.optim.SGD(snn.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()

    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()
        x = np.array(images)

        images = images.float().to(device)
        outputs = snn(images)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
    # loss_train_record.append('%.4f' % (running_loss/6))
    # running_loss = 0
    # print(loss_train_record)
    # threshold_history.append(snn.w.data.item())
    # print("threshold_history：", threshold_history)


    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)


    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    # print(acc_record)
    if epoch % 5 == 0:
        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
            # 'thresholding': snn.w.data.sigmoid().item()
            # 'threshold': snn.w.data.item()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc

# print('loss_train_record:', loss_train_record)
# print('acc_record:', acc_record)
# print('best_acc:', best_acc)
