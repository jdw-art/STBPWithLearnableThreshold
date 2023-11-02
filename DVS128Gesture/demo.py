import torch
import torchvision
from matplotlib import pyplot as plt
from spikingjelly.datasets import play_frame
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask, dvs128_gesture

# from DVS128Gesture.DVS128Gesture import DVS128Gesture
# from DVS128Gesture import *

device = torch.device("cuda:0")

batch_size = 16

root_dir = 'E:/Dataset/DVS128Gesture'


