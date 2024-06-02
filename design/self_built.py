#1.载入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms

BATCH_SIZE=16
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")#看自己的主机是用gpu还是cpu
EPOCHS=5#训练数据集的轮次
input_size=28#图像总的尺寸28*28
num_classes=10#标签种类数

#3.构建pipeline,对图像进行处理
pipeline=transforms.Compose([transforms.ToTensor(),#将图像转为tensor
        transforms.Normalize((0.1307,),(0.3081,)) #正则化，进行降低模型复杂度
    ])

#4.下载和加载数据
from torch.utils.data import DataLoader

#下载数据集
train_set=datasets.MNIST("data",train=True,download=False,transform=pipeline)

test_set=datasets.MNIST("data",train=False,download=False,transform=pipeline)

#加载数据集
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)


class Batch_CNN(nn.Module):  # 28*28
    def __init__(self, c):
        super(Batch_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(c, 24, kernel_size=3),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            #24*24
            nn.Conv2d(24, 48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            #12 * 12
            nn.Conv2d(48, 96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            # 10 * 10
            nn.Conv2d(96, 196, kernel_size=3),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(196*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 196),
            nn.ReLU(inplace=True),
            nn.Linear(196, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output