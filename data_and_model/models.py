import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -----------------------  MNIST  -------------------------

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(     
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc1 = nn.Linear(7*7*32,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,7*7*32)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return F.log_softmax(output,dim=1)

class MNIST_LR_Net(nn.Module):
    def __init__(self):
        super(MNIST_LR_Net, self).__init__()

        self.hidden1 = nn.Linear(28 * 28, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.hidden1(x), inplace=True) 
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x)

        return F.log_softmax(x, dim=1)

class MNIST_RNN(nn.Module):
    def __init__(self):
        super(MNIST_RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=28,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1,28,28)
        r_out, (h_n, h_c) = self.rnn(x, None)   

        out = self.out(r_out[:, -1, :])
        return out



# ----------------------  EMNIST  -------------------------

class EMNIST_CNN(nn.Module):
    def __init__(self):
        super(EMNIST_CNN,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(       
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc1 = nn.Linear(7*7*32,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,7*7*32)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return F.log_softmax(output,dim=1)



# ----------------------  CIFAR10  -------------------------

class CIFAR10_VGG9(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_VGG9, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._initialize_weights()

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, residual_block, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(residual_block, 64, 2, stride=1)
        self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
        self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
        self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        # out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def CIFAR10_ResNet18():
    return ResNet(ResidualBlock)



# ----------------------  CIFAR100  -------------------------

def CIFAR100_ResNet18():
    return ResNet(ResidualBlock,num_classes=100)




def get_model(dataset,model_name="CNN",batch=None):
    
    if dataset == "MNIST":
        if model_name == "CNN":
            model = MNIST_CNN()
        elif model_name == "MCLR":
            model = MNIST_LR_Net()
        elif model_name == "RNN":
            model = MNIST_RNN()
        else:
            model = "break"

    elif dataset == "EMNIST":
        model = EMNIST_CNN()

    elif dataset == "CIFAR10":
        if model_name == "CNN":
            model =  CIFAR10_VGG9()
        else:
            model =  CIFAR10_ResNet18()

    else:
        model = CIFAR100_ResNet18()

    return model
