import torch

# 加载.t7文件
# file_path = "../Original/checkpoint/ckptspiking_model.t7"
file_path = "E:\Homework\SNN_Project\STBPWithLearnableThreshold\Original\checkpoint\ckptspiking_model.t7"
# file_path = "../CIFAR10/checkpoint/spiking_cnn_model.t7"
data = torch.load(file_path)
# 输出内容
print(data)