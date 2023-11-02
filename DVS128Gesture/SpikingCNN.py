import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven import functional, layer, surrogate
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from torchvision import transforms


def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, use_max_pool, alpha_learnable,
                           detach_reset):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    # alpha_learnable是代理梯度中的可训练参数
    # 第一个下采样模块相当于网络中将输入转化为脉冲的编码过程
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


# 分类器模块中包含两层全连接神经元
def create_2fc(channels, h, w, dpp, class_num, init_tau, alpha_learnable, detach_reset):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        # 输出的特征维度缩小为输入的1/4
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
    )


class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x): # x初始形状为[N, T, 2, *, *]
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, *, *]，经过转换变为[T, N, 2, *, *]
        # 对输入的第一个时间步进行卷积操作，并通过全连接层和平均池化层得到输出脉冲计数。相当于编码操作
        out_spikes_counter = self.boost(self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            # 遍历剩余的时间步，对每个时间步进行卷积操作，并通过全连接层和平均池化层，将脉冲计数累加到 out_spikes_counter 中。
            out_spikes_counter += self.boost(self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter


class DVS128GestureNet(NeuromorphicNet):
    # DVS128Gesture网络的整体架构：{c128k3s1-BN-PLIF-MPk2s2}*5DP-FC512-PLIF-DP-FC110-PLIFAPk10s10
    def __init__(self, T, init_tau, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        # 调用父类构造函数，对参数进行初始化
        super().__init__(T=T, init_tau=init_tau, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable,
                         detach_reset=detach_reset)
        # 初始化输入图像的宽度和高度。
        w = 128
        h = 128
        # 创建卷积层序列
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable,
                                           detach_reset=detach_reset)
        # 创建全连接层序列
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=11,
                             init_tau=init_tau, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
