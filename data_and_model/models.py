import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from abc import ABC

# 定义三个模型，CNN，DNN，MCLR(多重逻辑回归）

class CNN(nn.Module):
    def __init__(self,in_dim,width,height,n_classes):   # 输入维度，宽度，高度，类别数
        super(CNN,self).__init__()
        self.width = width
        self.height =height

        self.conv1 = nn.Sequential(        # 卷积1，输出维度8，卷积核为3,步长为1,填充1
            nn.Conv2d(in_dim,8,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(        # 卷积2，输出维度16，卷积核为3,步长为1，填充1
            nn.Conv2d(8,16,3,1,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )

        self.fc1 = nn.Linear(((width//2)//2)*((height//2)//2)*16,128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84,n_classes)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,((self.width//2)//2)*((self.height//2)//2)*16)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = F.log_softmax(output,dim=1)
        return output


class DNN(nn.Module):
    def __init__(self,in_dim,width,height,n_classes,mid_dim=128):  # 输入维度，宽度，高度，类别数,中间维度默认128
        super(DNN,self).__init__()

        self.in_dim =in_dim
        self.width = width
        self.height =height

        self.fc1 = nn.Linear(in_dim * width * height,mid_dim)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self,x):
        x = x.view(-1,self.in_dim*self.height*self.width)
        output = self.fc1(x)
        x = F.relu(output)
        output = self.fc2(x)
        output = F.log_softmax(output,dim=1)
        return output


class MCLR_Logistic(nn.Module):
    def __init__(self, in_dim,width,height,n_classes):   # 输入维度，宽度，高度，类别数
        super(MCLR_Logistic, self).__init__()
        self.in_dim =in_dim
        self.width = width
        self.height =height
        self.fc1 = nn.Linear(in_dim * width * height,n_classes)

    def forward(self, x):
        x = x.view(-1,self.in_dim*self.height*self.width)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output



class CNNCIFAR(nn.Module):
    def __init__(self,in_dim,width,height,n_classes):   # 输入维度，宽度，高度，类别数
        super(CNNCIFAR,self).__init__()
        self.width = width
        self.height =height

        self.conv1 = nn.Sequential(        # 卷积1，输出维度6，卷积核为3,步长为1,填充1
            nn.Conv2d(in_dim,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(        # 卷积2，输出维度32，卷积核为3,步长为1，填充1
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.conv3 = nn.Sequential(        # 卷积2，输出维度128，卷积核为3,步长为1，填充1
            nn.Conv2d(32,128,3,1,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )

        self.fc1 = nn.Linear(4*4*128,256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84,n_classes)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        output = out_conv3.view(-1,16*128)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = F.log_softmax(output,dim=1)
        return output


# 返回数据的信息：尺寸，通道，标签数量
def get_model(dataset,model="CNN"):

    if dataset =='MNIST':
        width, height, dim, num_label = 28, 28, 1, 10
    elif dataset =='CIFAR10':
        width, height, dim, num_label = 32, 32, 3, 10
        return CNNCIFAR(dim, width, height, num_label)
    else:
        width, height, dim, num_label = 28, 28, 1, 62

    if model == "CNN":
        m = CNN(dim, width, height, num_label)
    elif model == "DNN":
        m = DNN(dim, width, height, num_label)
    else:
        m = MCLR_Logistic(dim, width, height, num_label)

    return m