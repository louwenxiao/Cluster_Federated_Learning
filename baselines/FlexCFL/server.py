import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import TruncatedSVD
from data_and_model.models import get_model
import sys

class server(object):
    def __init__(self,model,device,dataset,clients,args,eta_g=0):
        self.model = model.to(device)
        self.device = device
        self.test_data = dataset
        self.clients = clients
        self.args = args
        self.eta_g = eta_g

    # 普通的FedAvg用到这个函数，基于聚类的FedAvg用下面的函数
    def aggregate_model(self):
        model_states = []
        for i in self.clients_id:
            model_states.append(torch.load('./cache/model_state_{}.pt'.format(i)))

        global_model_state = copy.deepcopy(model_states[0])

        for key in global_model_state.keys():
            for i in range(1, len(model_states)):
                global_model_state[key] += model_states[i][key]
            global_model_state[key] = torch.div(global_model_state[key], len(model_states))

        self.model.load_state_dict(global_model_state)
        torch.save(self.model.state_dict(), './cache/global_model.pt')


    def gain_acc(self):
        
        test_correct = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = test_correct / len(self.test_data.dataset)
        
        return test_loss,test_acc


    # 组冷启动
    def Group_Cold_Start(self,k):
        # 广播全局模型到所有用户,所有用户预训练
        W = []
        torch.save(self.model.state_dict(), "./cache/global_model.pt")
        for client in self.clients:
            W.append(client.pre_train())
        
        delta_w = np.array(W) # shape=(n_clients, n_params)
        
        # Decomposed the directions of updates to num_group of directional vectors
        svd = TruncatedSVD(n_components=k)
        decomp_updates = svd.fit_transform(delta_w.T) # shape=(n_params, n_groups)
        # n_components = decomp_updates.shape[-1]

        decomposed_cossim_matrix = cosine_similarity(delta_w, decomp_updates.T) # shape=(n_clients, n_clients)
        
        affinity_matrix = decomposed_cossim_matrix
        result = KMeans(k, max_iter=100).fit(affinity_matrix)

        self.cluster = [[] for i in range(k)]       # 存放每个簇对应的用户下标
        cluster = result.labels_
        for i in range(len(self.clients)):
            index = cluster[i]
            self.cluster[index].append(i)
        self.cluster_model = []
        for i in range(k):
            self.cluster_model.append(get_model(self.args.dataset))        # 初始化k个簇模型,并送给簇内部每一个用户
        
        self.Clients_get_model()                # 更新每个用户的本地模型
        print(self.cluster)



    # 组内并行训练
    def IntraGroupUpdate(self):
        
        # 组内更新，获得中间模型enumerate
        client_models = []
        for i in range(len(self.clients)):
            m = torch.load('./cache/model_state_{}.pt'.format(i))
            client_models.append(copy.deepcopy(m))

        self.W_ = [copy.deepcopy(self.model) for _ in range(len(self.cluster_model))]
        for index,model in enumerate(self.W_):
            m = copy.deepcopy(model.state_dict())
            for key in m.keys():
                m[key] = m[key]*0.0         # 参数清零
                for i in self.cluster[index]:
                    m[key] += client_models[i][key]
                m[key] = torch.div(m[key],len(self.cluster[index])) 
            self.W_[index].load_state_dict(m)


    # 组之间聚合
    def InterGroupAggregation(self):
        delta_W = [copy.deepcopy(self.model) for _ in range(len(self.cluster_model))]

        m = copy.deepcopy(self.model.state_dict())
        
        param = L2_norm(self.W_)                          # 计算每个模型的二范数
        print(param)

        for index in range(len(self.cluster_model)):

            for key in delta_W[index].state_dict().keys():
                m[key] = m[key] *0.0
                for i in range(len(self.cluster_model)):
                    if i == index:
                        continue
                    m[key] = m[key] + torch.div(self.eta_g * self.W_[i].state_dict()[key],param[i])
                #delta_W[index].state_dict()[key] = torch.div(delta_W[index].state_dict()[key],len(self.cluster_model)-1)

                m[key] = self.W_[index].state_dict()[key] + m[key]

            torch.save(m, './cache/global_model_{}.pt'.format(index))
            self.cluster_model[index].load_state_dict(m)
    


    # 簇内用户获得模型
    def Clients_get_model(self):
        for i,cluster in enumerate(self.cluster):
            torch.save(self.cluster_model[i].state_dict(), './cache/global_model_{}.pt'.format(i))
            for j in cluster:
                self.clients[j].get_cluster_model(i)



# 将模型变成一维向量，计算2范数
def L2_norm(models):
    param = [0 for i in range(len(models))]
    model_list = []
    for i in range(len(models)):
        for key in models[i].state_dict().keys():
            model_list.extend(models[i].state_dict()[key].view(-1).tolist())
        param[i] = torch.norm(torch.tensor(model_list),p=2).item()
    
    return param
        