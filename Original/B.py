import torch

# 创建一个包含5个张量的列表，每个张量形状为（128，1，28，28）
tensor_list = [torch.randn(1, 28, 28) for _ in range(5)]

# 使用torch.stack将它们堆叠成形状为（128，5，28，28）的张量
stacked_tensor = torch.stack(tensor_list, dim=0)

print(stacked_tensor.shape)  # 输出：torch.Size([128, 5, 28, 28])
