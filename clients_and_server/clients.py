# 需要用到的函数：train，test，加载全局模型(查看全局模型是否存在）并更新，保存模型
# 输入的参数：本地轮数，用户id，学习率，一批的大小，训练测试数据
# 然后根据 模型、数据集、学习率、本地迭代次数、优化器、用户id  创建用户 global_nums个用户
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import copy


class client(object):
    def __init__(self,id,model,device,dataset,learning_rate,optimizer,local_epochs,server_id):
        self.id = id
        self.device = device
        self.server_id = server_id
        self.model= model.to(device)
        self.model_init = model
        self.train_data = dataset[0]
        self.test_data = dataset[1]
        self.lr = learning_rate
        self.optimizer = optimizer
        self.epochs = local_epochs

    # 获得簇模型
    def get_cluster_modal(self):
        model = copy.deepcopy(self.model_init)
        model.load_state_dict(torch.load('./cache/global_model_{}.pt'.format(self.server_id)))
        self.model = copy.deepcopy(model)

    
    # 全局模型更新，每个用户载入新的模型
    def get_model(self):
        model = copy.deepcopy(self.model_init)
        model.load_state_dict(torch.load('./cache/global_model.pt'))
        self.model = copy.deepcopy(model)


    # 本地进行训练
    def local_train(self):
        self.model.train()

        if self.optimizer == "SGD":
            optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            optimizer = optim.Adam(params=self.model.parameters(),lr=self.lr)

        for i in range(self.epochs):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), './cache/model_state_{}.pt'.format(self.id))


    # 在聚类前的预训练
    def pre_train(self,epoch=10):
        self.model.train()

        if self.optimizer == "SGD":
            optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            optimizer = optim.Adam(params=self.model.parameters(),lr=self.lr)

        for i in range(epoch):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), './cache/model_state_{}.pt'.format(self.id))


    # 模型测试
    def test_model(self):
        test_loss = 0
        test_correct = 0
        model = copy.deepcopy(self.model)
        model.eval()
        
        with torch.no_grad():   # torch.no_grad()是一个上下文管理器，用来禁止梯度的计算
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                output = model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                test_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()

        return test_loss, test_correct / len(self.test_data.dataset)


    # 更新用户自己所属于的簇
    def updata_clu(self,id):
        self.server_id = id
