# 需要用到的函数：train，test，加载全局模型(查看全局模型是否存在）并更新，保存模型
# 输入的参数：本地轮数，用户id，学习率，一批的大小，训练测试数据
# 然后根据 模型、数据集、学习率、本地迭代次数、优化器、用户id  创建用户 global_nums个用户
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import copy
import sys


class client(object):
    def __init__(self,id,model,device,dataset,args):
        self.id = id
        self.device = device
        self.model = model.to(device)
        #print(next(self.model.parameters()).device)
        self.train_data = dataset[0]
        self.test_data = dataset[1]

        self.optimizer = self.__initial_optimizer(args.optimizer,args.lr,args.momentum)
        self.scheduler_lr = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)
        self.epochs = args.local_epochs


    def __initial_optimizer(self,optimizer,lr,momentum):             # 初始化优化器
        if optimizer == "SGD":
            opt = optim.SGD(params=self.model.parameters(), lr=lr, momentum=momentum)
        else:
            opt = optim.Adam(params=self.model.parameters(),lr=lr)

        return opt
    

    # 全局模型更新，每个用户载入新的模型
    def get_model(self,model):
        self.model = copy.deepcopy(model)

    
    # 用户获得簇模型
    def get_cluster_model(self,id):
        self.model.load_state_dict(torch.load('./cache/global_model_{}.pt'.format(id)))


    # 本地进行训练
    def local_train(self):
        self.model.train()

        for i in range(self.epochs):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                self.optimizer.step()
        
        self.scheduler_lr.step()            # 更新学习率

        torch.save(self.model.state_dict(), './cache/model_state_{}.pt'.format(self.id))
        


    # 模型测试
    def test_model(self):
        test_loss = 0
        test_correct = 0
        
        with torch.no_grad():   # torch.no_grad()是一个上下文管理器，用来禁止梯度的计算
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                output = self.model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                test_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()

        return test_loss, test_correct / len(self.test_data.dataset)


    def pre_train(self,epoch=1):
        # 首先获取全局模型,model_copy存放预训练的结果，返回梯度
        self.model.load_state_dict(torch.load('./cache/global_model.pt'))
        self.model.to(self.device)
        model_copy = copy.deepcopy(self.model)
        optimizer = optim.SGD(model_copy.parameters(), lr=0.01, momentum=0.9)
        # 预训练
        model_copy.train()
        for i in range(epoch):
            for data,target in self.train_data:
                data,target = Variable(data).to(self.device),Variable(target).to(self.device)

                self.optimizer.zero_grad()
                output = model_copy(data)

                loss = nn.CrossEntropyLoss()(output,target)
                loss.backward()

                optimizer.step()

        # 返回模型梯度,生成一维数据
        model_list = []
        for key in model_copy.state_dict().keys():
            data = model_copy.state_dict()[key] - self.model.state_dict()[key]
            model_list.extend(data.view(-1).tolist())
        
        return model_list
