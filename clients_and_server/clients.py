# 需要用到的函数：train，test，加载全局模型(查看全局模型是否存在）并更新，保存模型
# 输入的参数：本地轮数，用户id，学习率，一批的大小，训练测试数据
# 然后根据 模型、数据集、学习率、本地迭代次数、优化器、用户id  创建用户 global_nums个用户
from data_and_model.models import get_model
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import copy
import sys
import time


class get_client(object):
    def __init__(self,id,model,device,dataset,args):
        self.id = id
        self.device = device
        self.args = args
        # self.global_or_cluster_model = copy.deepcopy(get_model(dataset=args.dataset).to(device))    # 簇模型或者是全局模型，计算梯度使用
        self.global_or_cluster_model = copy.deepcopy(model.to(device))
        self.train_data = dataset[0]
        self.test_data = dataset[1]

        # self.model = copy.deepcopy(model.to(device))        # 用户自己的模型
        self.model = copy.deepcopy(get_model(dataset=args.dataset,model_name=args.model_name).to(device))
        self.model.load_state_dict(model.state_dict())
        self.optimizer = self.__initial_optimizer(args.optimizer,self.model,args.lr,args.momentum)
        self.scheduler_lr = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)

    def __initial_optimizer(self,optimizer,model,lr,momentum):             # initialize optimizer
        if optimizer == "SGD":
            opt = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
        else:
            opt = optim.Adam(params=model.parameters(),lr=lr)
        return opt


    # 在聚类前的预训练
    def pre_train(self,epoch=1):
        grad_model = copy.deepcopy(self.global_or_cluster_model.to(self.device))
        
        # def weigth_init(m):         # 针对CNN或者MCLR模型初始化
        #     if isinstance(m, nn.Linear):    # 全连接层初始化
        #         m.weight.data.zero_()
        #         m.bias.data.zero_()
        #     if isinstance(m, nn.Conv2d):    # 卷积层初始化
        #         nn.init.constant_(m.weight, 0.)
        #         nn.init.constant_(m.bias, 0.)
        # grad_model.apply(weigth_init)

        # param = copy.deepcopy(grad_model.state_dict())        # 针对resnet模型初始化
        # for key in param.keys():
        #     param[key] = param[key]*0
        # grad_model.load_state_dict(param)

        # for name, param in grad_model.named_parameters():     # 针对RNN模型进行初始化
        #     nn.init.zeros_(param)

        # print(grad_model.state_dict())

        grad_model.train()
        initial_model = copy.deepcopy(grad_model.state_dict())

        optimizer = optim.SGD(params=grad_model.parameters(), lr=0.05, momentum=0.9)
        for _ in range(epoch):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                optimizer.zero_grad()
                output = grad_model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

        grad_model = copy.deepcopy(grad_model.state_dict())

        for key in grad_model.keys():
            grad_model[key] = grad_model[key] - initial_model[key]
        torch.save(grad_model, './cache/grad_model_{}.pt'.format(self.id))


    # 本地进行训练
    def local_train(self):
        grad_model = copy.deepcopy(self.model.state_dict())     # 每一轮训练后，获得梯度，用于划分簇聚合模型
        self.model.train()

        for _ in range(self.args.local_epochs):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                self.optimizer.step()
        
            self.scheduler_lr.step()

        for key in grad_model.keys():
            grad_model[key] = self.model.state_dict()[key] - grad_model[key]
        torch.save(grad_model, './cache/grad_model_{}.pt'.format(self.id))
        torch.save(self.model.state_dict(), './cache/client_model_{}.pt'.format(self.id))


    # 模型测试
    def test_model(self):
        test_loss = 0
        test_correct = 0
        self.model.eval()

        with torch.no_grad():   # torch.no_grad()是一个上下文管理器，用来禁止梯度的计算
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                output = self.model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                test_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()

        return test_loss/len(self.test_data.dataset), test_correct / len(self.test_data.dataset)


    def get_global_model(self):
        self.model.load_state_dict(torch.load('./cache/global_model.pt'))

    def get_cluster_model(self,clients_id):
        cluster = 0
        for cluster,clients in enumerate(clients_id):
            if self.id in clients:
                self.model.load_state_dict(torch.load('./cache/cluster_model_{}.pt'.format(cluster)))



