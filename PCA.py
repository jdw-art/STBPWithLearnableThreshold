from sklearn.decomposition import PCA
# import numpy as np
# X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.transform(X))
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt

from spikingjelly.activation_based import encoding
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 加载MNIST数据集
mnist_dataset = datasets.MNIST(root='E:\Dataset\MNIST', train=True, download=True, transform=transforms.ToTensor())

# 定义泊松编码器
encoder = encoding.PoissonEncoder()

# 选取十张图像并进行泊松编码
selected_labels = list(range(10))  # 选取的标签列表
encoded_images = []
images = []
labels = []
train_data = []
test_data = []

for i, (img, label) in enumerate(mnist_dataset):
    encoded_img = encoder(img).squeeze().numpy()
    encoded_images.append((encoded_img, label))
    images.append(encoded_images[i][0])
    labels.append(encoded_images[i][1])

images = torch.tensor(images)
labels = torch.tensor(labels)

for i in range(0, 10):
    # 指定要选择的标签
    target_label = i
    # 找到与指定标签相同的图像的索引
    indices = (labels == target_label).nonzero(as_tuple=True)[0]
    # 随机选择一个索引
    selected_index = indices[0]
    # 根据选中的索引获取图像数据
    selected_image = images[selected_index]
    encode_image = selected_image.view(1, -1).numpy()
    train_data.append(encode_image)

    # 找到与指定标签相同的图像的索引
    indices = np.where(labels == target_label)[0]
    # 随机选择十个索引
    selected_indices = np.random.choice(indices, size=10, replace=False)
    # 根据选中的索引获取图像数据
    selected_images = images[selected_indices]
    selected_images = np.array(selected_images).reshape(10, 784)
    test_data.append(selected_images)

train_data = np.array(train_data).reshape(10, 784)
test_data = np.array(test_data).reshape(100, 784)

pca = PCA(n_components=100)
pca.fit(test_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)

print(train_data, test_data)


