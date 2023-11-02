import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
thresh = 0.5 # neuronal threshold 神经元阈值
lens = 0.5 # hyper-parameters of approximate function 超参数

# 代理梯度参数learnable a
init_a = 1

decay = 0.2 # decay constants 衰减常数
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
drop_rate = 0.5
kappa = 0.5 # parameter of learnable thresholding

num_epochs = 100 # max epoch迭代次数
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thre):
        ctx.save_for_backward(input)
        ctx.threshold = thre
        return input.gt(thre).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        thre = ctx.threshold
        grad_input = grad_output.clone()
        temp = abs(input - thre) < lens
        return grad_input * temp.float(), None

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike, thre):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem, thre) # act_fun : approximation firing function
    return mem, spike

class LIFNeuron(nn.Module):
    def __init__(self):
        super(LIFNeuron, self).__init__()
        init_w = kappa
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        self.threshold_history = []

    def forward(self, ops, x, membrane_potential, out):
        membrane_potential, out = mem_update(ops, x, membrane_potential, out, self._threshold())
        return membrane_potential, out

    # 通过sigmoid函数计算threshold
    def _threshold(self):
        threshold = self.w.data.sigmoid().item()
        self.threshold_history.append(threshold)  # 保存历史阈值
        return threshold

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [128, 10]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def _dropout(x, total_in, total_out):
    scale = total_out / total_in
    scale_mean = torch.mean(scale, dim=1)
    p = (scale - scale_mean.min()) / (scale.max() - scale.min())
    p = 1 - (1 - p) * drop_rate / (1 - p.mean())
    p = torch.clamp(p, min=0., max=1.)
    d = torch.bernoulli(p).to(x.device)
    drop = d.clone().detach().to(x.device)

    return x * drop

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        init_w = kappa
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

        # 对卷积层和全连接层的权重和阈值进行初始化
        for m in self.modules():

            if isinstance(m, nn.Conv2d):  # 对卷积层中的卷积核进行正则化
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                # 权重初始化，突触权重采用零均值的高斯随机分布和√(κ/nl)（nl：扇入突触的数量）的标准差初始化
                variance1 = math.sqrt(2. / n)
                m.weight.data.normal_(0, variance1)
                # m.threshold = self._threshold()

            elif isinstance(m, nn.Linear):  # 对全连接层中的权重进行正则化
                size = m.weight.size()
                fan_in = size[1]
                # 权重初始化，突触权重采用零均值的高斯随机分布和√(κ/nl)（nl：扇入突触的数量）的标准差初始化
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                # m.threshold = self._threshold()

    def _threshold(self):
        return self.w.data.sigmoid().item()

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        # 输出层之前一层的全连接层神经元的累计脉冲流，形状为（batch_size, 128）
        Total_f1_output = torch.zeros(input.size(0), 128, device=input.device)
        LIF_in_f1 = torch.zeros(input.size(0), 128, device=input.device)
        LIF_Neuron = LIFNeuron()

        for step in range(time_window): # simulation time steps
            # x = input > torch.rand(input.size(), device=device) # prob. firing
            x = input

            c1_mem, c1_spike = LIF_Neuron.forward(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = LIF_Neuron.forward(self.conv2,x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            in_layer = self.fc1(x)
            h1_mem, h1_spike = LIF_Neuron.forward(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            # h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            # h2_sumspike += h2_spike

            h2_mem, h2_spike = LIF_Neuron.forward(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike

            # LIF_in_f1 += in_layer
            # Total_f1_output += h1_spike
            # h1_sumspike += h1_spike

            # if self.training:
            #     h1_spike = _dropout(h1_spike, LIF_in_f1, Total_f1_output)

            # h2_mem = h2_mem + self.fc2(h1_spike)

        outputs = h2_sumspike / time_window
        # outputs = h2_mem / time_window / self.fc2.threshold

        return outputs