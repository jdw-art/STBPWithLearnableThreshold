from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
import torch.optim as optim

from tqdm import tqdm
from SpikingCNN import*
# from SpikingVGG9 import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


names = 'spiking_cnn_model'
data_path = 'E:\Dataset\CIFAR-10' # input your path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
print('==> Preparing data..')
# 数据增强操作：
# 随机裁剪（RandomCrop）：通过transforms.RandomCrop函数实现，可以指定裁剪的大小和填充。
# 随机水平翻转（RandomHorizontalFlip）：通过transforms.RandomHorizontalFlip函数实现，以一定的概率对图像进行水平翻转。
# 随机垂直翻转（RandomVerticalFlip）：通过transforms.RandomVerticalFlip函数实现，以一定的概率对图像进行垂直翻转。
# 随机旋转（RandomRotation）：通过transforms.RandomRotation函数实现，可以指定旋转的角度范围。
# 随机亮度调整（RandomBrightness）：通过transforms.ColorJitter函数实现，可以在一定范围内调整图像的亮度。
# 随机对比度调整（RandomContrast）：通过transforms.ColorJitter函数实现，可以在一定范围内调整图像的对比度。
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(degrees=(-15, 15)),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last =True )

testset = torchvision.datasets.CIFAR10(root= data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last =True)

net = SCNN()
net = net.to(device)

criterion = nn.MSELoss() # Mean square error loss
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
optimizer = assign_optimizer(net, lrs=learning_rate)
# optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# 加上L2正则化weight_decay=0.0001
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.0001)
# using SGD+CosineAnnealing could achieve better results
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    loop = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in loop:
        inputs  = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.cpu().max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        if batch_idx%200==0:
            elapsed = time.time() -starts
            print(batch_idx,'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                       %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Time past: ', elapsed, 's', 'Iter number:', epoch)
    loss_train_record.append(train_loss)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    loop = tqdm(enumerate(testloader), total=len(testloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in loop:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader),'Loss: %.5f | Acc: %.5f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        loss_test_record.append(test_loss)


    # Save checkpoint.
    acc = 100.*correct/total
    acc_record.append(acc)


    if best_acc<acc:
        best_acc = acc
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
            'loss_train_record': loss_train_record,
            'loss_test_record': loss_test_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + names + '.t7')


for epoch in range(start_epoch, start_epoch+num_epochs):

    starts = time.time()
    train(epoch)
    test(epoch)
    elapsed =  time.time() - starts
    optimizer = lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=35)
    print (" \n\n\n\n\n\n\n")
    print('Time past: ',elapsed,'s', 'Iter number:', epoch)