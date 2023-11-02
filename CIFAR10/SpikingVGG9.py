import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyper parameters
# thresh = 0.75
lens = 0.5
probs = 0.5

drop_rate = 0.5

alpha = 0.85
decay = 0.25
batch_size = 20 # increasing batch_size windows can help performance
num_epochs = 100
learning_rate = 0.1 # learning rate
time_window = 10  # increasing sampling windows can help performance

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
def mem_update(conv, x, mem, spike, thres):
    mem = mem * decay * (1. - spike) + conv(x)
    spike = act_fun(mem, thres)
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


def Pooling_sNeuron(membrane_potential, threshold, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = act_fun(ex_membrane, threshold)

    return membrane_potential, out

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
weights = torch.zeros(cfg_fc[-1], 10, device=device,requires_grad = False)  # cfg_fc[-1]
vote_num = cfg_fc[-1] // 10
for i in range(cfg_fc[-1]):
    weights.data[i][i // vote_num] = 10 / cfg_fc[-1]


def assign_optimizer(model, lrs=1e-3):
    rate = 1e-1
    fc0_params = list(map(id, model.fc0.parameters()))
    fc1_params = list(map(id, model.fc1.parameters()))
    base_params = filter(lambda p: id(p) not in fc0_params + fc1_params, model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc0.parameters(), 'lr': lrs * rate},
        {'params': model.fc1.parameters(), 'lr': lrs * rate}, ]
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


class SCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()

        self.cnn11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.cnn31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.fc0 = nn.Linear(256 * 4 * 4, 1024, bias=False)
        self.fc1 = nn.Linear(1024, 10, bias=False)

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

        mem_11 = spike_11 = torch.zeros(input.size(0), 64, 32, 32, device=device)
        mem_12 = spike_12 = torch.zeros(input.size(0), 64, 32, 32, device=device)
        mem_1s = spike_1s = torch.zeros(input.size(0), 64, 16, 16, device=device)

        mem_21 = spike_21 = torch.zeros(input.size(0), 128, 16, 16, device=device)
        mem_22 = spike_22 = torch.zeros(input.size(0), 128, 16, 16, device=device)
        mem_2s = spike_2s = torch.zeros(input.size(0), 128, 8, 8, device=device)

        mem_31 = spike_31 = torch.zeros(input.size(0), 256, 8, 8, device=device)
        mem_32 = spike_32 = torch.zeros(input.size(0), 256, 8, 8, device=device)
        mem_33 = spike_33 = torch.zeros(input.size(0), 256, 8, 8, device=device)
        mem_3s = spike_3s = torch.zeros(input.size(0), 256, 4, 4, device=device)

        membrane_f0 = spike_f0 = torch.zeros(input.size(0), 1024, device=input.device)
        membrane_f1 = spike_f1 = sumspike_f1 = torch.zeros(input.size(0), 10, device=input.device)

        LIF_Neuron = LIFNeuron()

        for step in range(time_window):
            # x = input > torch.rand(input.size(), device=device)
            x = input

            mem_11, spike_11 = LIF_Neuron.forward(self.cnn11, x.float(), mem_11, spike_11)
            mem_12, spike_12 = LIF_Neuron.forward(self.cnn12, F.dropout(spike_11, p=probs, training=self.training), mem_12, spike_12)
            mem_1s = mem_1s + self.avgpool1(spike_12)
            mem_1s, spike_1s = Pooling_sNeuron(mem_1s, 0.5, step)

            mem_21, spike_21 = LIF_Neuron.forward(self.cnn21, spike_1s, mem_21, spike_21)
            mem_22, spike_22 = LIF_Neuron.forward(self.cnn22, F.dropout(spike_21, p=probs, training=self.training), mem_22, spike_22)
            mem_2s = mem_2s + self.avgpool2(spike_22)
            mem_2s, spike_2s = Pooling_sNeuron(mem_2s, 0.5, step)

            mem_31, spike_31 = LIF_Neuron.forward(self.cnn31, spike_2s, mem_31, spike_31)
            mem_32, spike_32 = LIF_Neuron.forward(self.cnn32, F.dropout(spike_31, p=probs, training=self.training), mem_32, spike_32)
            mem_33, spike_33 = LIF_Neuron.forward(self.cnn33, F.dropout(spike_32, p=probs, training=self.training), mem_33, spike_33)
            mem_3s = mem_3s + self.avgpool3(spike_33)
            mem_3s, spike_3s = Pooling_sNeuron(mem_3s, 0.5, step)

            x = spike_3s.view(batch_size, -1)

            membrane_f0, spike_f0 = LIF_Neuron.forward(self.fc0, F.dropout(x, p=probs, training=self.training), membrane_f0, spike_f0)
            membrane_f1, spike_f1 = LIF_Neuron.forward(self.fc1, spike_f0, membrane_f1, spike_f1)

            sumspike_f1 += spike_f1

        outputs = sumspike_f1 / time_window
        # outputs = outputs.mm(weights)

        return outputs
