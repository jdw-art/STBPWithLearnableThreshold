import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyper parameters
# thresh = 0.75 # 0.5 in MNIST
lens = 0.5
probs = 0.5

drop_rate = 0.5

alpha = 0.85
decay = 0.25
batch_size = 20 # increasing batch_size windows can help performance
num_epochs = 50
learning_rate = 0.1 # 0.001 in MNIST
time_window = 10  # increasing sampling windows can help performance 20 in MNIST

kappa = 0.5 # parameter of learnable thresholding


class ActFun(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input, thres):
        ctx.save_for_backward(input)
        ctx.threshold = thres
        return input.gt(thres).float()

    @staticmethod
    def backward(ctx, grad_output):
        # au function
        input, = ctx.saved_tensors
        thre = ctx.threshold
        grad_input = grad_output.clone()
        temp = abs(input - thre) < lens
        return grad_input * temp.float(), None


# membrane potential update
# 使用硬重置
# def mem_update(conv, x, mem, spike, thres):
#     mem = mem * decay * (1. - spike) + conv(x)
#     spike = act_fun(mem, thres)
#     return mem, spike


# 使用软重置
def mem_update(conv, x, mem, spike, thres):
    mem = mem * decay + conv(x)
    spike = act_fun(mem, thres)
    mem = mem - spike
    return mem, spike

act_fun = ActFun.apply


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

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}'

# cnn layer :(in_plane,out_plane, stride,padding, kernel_size)
cfg_cnn = [(3, 128, 1, 1, 3),

           (128, 256, 1, 1, 3),

           (256, 512, 1, 1, 3),

           (512, 1024, 1, 1, 3),

           (1024, 512, 1, 1, 3),
           ]

cfg_kernel = [32, 32, 16, 8, 8, 8]
# fc layer
cfg_fc = [1024, 512, 100]

# voting matrix
# 通过投票矩阵，模型的输出特征将被平均分配到不同的分类标签上，从而实现了多模型集成或多次投票的效果
weights = torch.zeros(cfg_fc[-1], 10, device=device,requires_grad = False)  # cfg_fc[-1]
vote_num = cfg_fc[-1] // 10
for i in range(cfg_fc[-1]):
    weights.data[i][i // vote_num] = 10 / cfg_fc[-1]


def assign_optimizer(model, lrs=1e-3):
    rate = 1e-1
    fc1_params = list(map(id, model.fc1.parameters()))
    fc2_params = list(map(id, model.fc2.parameters()))
    fc3_params = list(map(id, model.fc3.parameters()))
    base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params, model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc1.parameters(), 'lr': lrs * rate},
        {'params': model.fc2.parameters(), 'lr': lrs * rate},
        {'params': model.fc3.parameters(), 'lr': lrs * rate}, ]
        , lr=lrs, momentum=0.9)
    print('successfully reset lr')
    return optimizer


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

# CIFARNet: 128C3-256C3-P2-512C3-P2-1024C3-512C3-1024-512
class SCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

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

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)

        LIF_Neuron = LIFNeuron()

        for step in range(time_window):
            x = input > torch.rand(input.size(), device=device)
            # x = input

            c1_mem, c1_spike = LIF_Neuron.forward(self.conv1, x.float(), c1_mem, c1_spike)

            c2_mem, c2_spike = LIF_Neuron.forward(self.conv2, F.dropout(c1_spike, p=probs, training=self.training), c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = F.dropout(x, p=probs, training=self.training)

            c3_mem, c3_spike = LIF_Neuron.forward(self.conv3, x, c3_mem, c3_spike)

            x = F.avg_pool2d(c3_spike, 2)
            x = F.dropout(x, p=probs, training=self.training)

            c4_mem, c4_spike = LIF_Neuron.forward(self.conv4, x, c4_mem, c4_spike)

            x = F.dropout(c4_spike, p=probs, training=self.training)
            c5_mem, c5_spike = LIF_Neuron.forward(self.conv5, x, c5_mem, c5_spike)
            x = c5_spike.view(batch_size, -1)

            h1_mem, h1_spike = LIF_Neuron.forward(self.fc1, F.dropout(x, p=probs, training=self.training), h1_mem, h1_spike)

            h2_mem, h2_spike = LIF_Neuron.forward(self.fc2, F.dropout(h1_spike, p=probs, training=self.training), h2_mem,h2_spike)

            h3_mem, h3_spike = LIF_Neuron.forward(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike
        outputs = h3_sumspike / time_window
        outputs = outputs.mm(weights)

        return outputs
