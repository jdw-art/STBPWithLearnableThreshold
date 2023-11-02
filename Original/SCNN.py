import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold 神经元阈值
lens = 0.5 # hyper-parameters of approximate function 超参数
decay = 0.2 # decay constants 衰减常数
num_classes = 10
batch_size  = 12
learning_rate = 1e-3

num_epochs = 100 # max epoch迭代次数
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    # soft reset
    # mem = decay * (mem - spike) + ops(x)
    # mem = mem * decay * (1. - 0.9 * spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

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

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

        # # 对卷积层和全连接层的权重和阈值进行初始化
        # for m in self.modules():
        #
        #     if isinstance(m, nn.Conv2d):  # 对卷积层中的卷积核进行正则化
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         # 权重初始化，突触权重采用零均值的高斯随机分布和√(κ/nl)（nl：扇入突触的数量）的标准差初始化
        #         variance1 = math.sqrt(2. / n)
        #         m.weight.data.normal_(0, variance1)
        #         # m.threshold = thresh
        #
        #     elif isinstance(m, nn.Linear):  # 对全连接层中的权重进行正则化
        #         size = m.weight.size()
        #         fan_in = size[1]
        #         # 权重初始化，突触权重采用零均值的高斯随机分布和√(κ/nl)（nl：扇入突触的数量）的标准差初始化
        #         variance2 = math.sqrt(2.0 / fan_in)
        #         m.weight.data.normal_(0.0, variance2)
        #         # m.threshold = thresh

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

            # h2_mem = h2_mem + self.fc2(h1_spike)

        outputs = h2_sumspike / time_window
        # outputs = h2_mem / time_window / thresh
        return outputs
