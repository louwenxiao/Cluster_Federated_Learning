import numpy as np
from sklearn.cluster import KMeans
import time
import random
import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import sys
# 三种聚类算法都算一下

def K_means_cluster(n_clients,k_means):
    clients = load_clients(n_clients=n_clients)
    a = torch.transpose(torch.tensor(clients),0,1)      # 矩阵转置
    u,s,v = torch.svd_lowrank(a,q=50)            # 主成分分析，使用奇异值分解
    
    print(u.shape)
    print(s.shape)
    print(v.shape)
    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()


    initial = random.sample(range(0, n_clients), k_means)  # 随机抽取k个中心
    init_model = [clients[i] for i in initial]
    indexes2 = [[] for i in range(k_means)]  # 保存一个副本
    print("init_model:",initial)
    print("shape:",np.array(clients).shape)
    num = 1

    while True:
        clusters = [[] for i in range(k_means)]    # 存放簇
        indexes = [[] for i in range(k_means)]      # 存放用户id

        for i in range(n_clients):
            # 计算这个用户与所有簇的距离，选择一个最小的距离
            distance = []
            for j in range(k_means):
                a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(init_model[j]))))
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)

        # 计算每个簇的均值
        print(np.array(clusters).shape)
        for i in range(k_means):
            print(i)
            a = np.array(clusters[i][0])*0.0
            for j in clusters[i]:
                a = np.array(a) + np.array(j)
            a = np.array(a) / len(clusters[i])
            init_model[i] = a.tolist()

        print(num, indexes, indexes2,'\n')
        # 判断返回的条件，如果100轮循环还没有返回，退出
        if (np.array(indexes).shape == np.array(indexes2).shape) and (np.array(indexes) == np.array(indexes2)).all():
            if num % 100 == 0:
                print("无法收敛，返回")
                return indexes
                
            break
        else:
            num = num + 1
            indexes2 = copy.deepcopy(indexes)

    return indexes

def Kmeans(n_clients):
    clients = load_clients(n_clients=n_clients)
    u,s,v = torch.svd_lowrank(torch.tensor(clients),q=100)            # 主成分分析，使用奇异值分解
    
    print(u.shape)
    print(s)
    print(v.shape)
    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()

    for k in range(len(s)):             # 选取主成分的90%的前k项
        if torch.sum(s[0:k])/torch.sum(s) >= 0.9:
            break
    print(k)               # 打印簇个数

    result = KMeans(k,max_iter=100).fit(clients).labels_    # kmeans++
    print(result)
    cluster = [[] for _ in range(k)]
    for i,index in enumerate(result):
        cluster[index].append(i)

    return cluster

def Hierarchical_clustering(n_clients,k_means):
    clients = load_clients(n_clients=n_clients)

    C = []
    for j in range(n_clients):
        C.append([[j],[clients[j]]])

    while len(C)>k_means:
        # 找出距离最近的两个聚类簇，合并
        M = [[0 for i in range(len(C))] for i in range(len(C))]
        for i in range(len(C)):
            for j in range(i+1,len(C)):
                pass
                M[i][j] = 0
                M[j][i] = M[i][j]

        min_x = 0
        min_y = 1
        min_dis = M[min_x][min_y]
        for i in range(len(C)):
            for j in range(i+1,len(C)):
                if min_dis < M[i][j]:
                    min_x = i
                    min_y = j
                    min_dis = M[min_x][min_y]

        # 合并距离最小的两个簇,删除另一个簇
        C[min_x][0].extend(C[min_y][0])
        C[min_x][1].extend(C[min_y][1])
        del C[min_y]

    indexes = [C[i][0] for i in range(k_means)]
    return indexes

def Kmeans_plusplus(n_clients,device,epoch):
    clients = load_clients(n_clients=n_clients)
    clients = torch.tensor(clients).to(device)

    distance = Distance_matrix(n_clients, clients)      # 计算距离矩阵
    clients = torch.tensor(distance)

    k = SVD_Dis(n_clients, clients, epoch)

    initial_model = [clients[0]]      # 初始化第一个簇
    # print(initial_model[0])
    while len(initial_model) < k:
        distances = []              # 保存最小距离
        for c in clients:
            min_dis = 100000        # 初始化一个最小距离
            for i in initial_model:
                dis = torch.norm(i-c,p=2,dim=0).item()
                if dis < min_dis:
                    min_dis = dis   # 更新最小距离      
            distances.append(min_dis)
        # print(distances)
        max_index = distances.index(max(distances))  
        distances[max_index] = 0        # 找到最大值的索引并置为0，防止噪声的问题

        max_index = distances.index(max(distances))  
        initial_model.append(clients[max_index])

    num = 1
    indexes2 = [[] for i in range(k)]  # 保存一个副本
    while True:
        clusters = [[] for _ in range(k)]    # 存放簇
        indexes = [[] for _ in range(k)]      # 存放用户id

        for i in range(n_clients):
            # 计算这个用户与所有簇的距离，选择一个最小的距离
            distance = []
            for j in range(k):
                # a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(initial_model[j]))))
                a = torch.norm(clients[i]-initial_model[j],p=2,dim=0).item()
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)

        # 计算每个簇的均值
        for i in range(k):
            a = clusters[i][0]*0.0
            for j in clusters[i]:
                a += j
            a = torch.div(a,len(clusters[i]))
            initial_model[i] = a

        # print(num, indexes, indexes2,'\n')
        # 判断返回的条件，如果100轮循环还没有返回，退出
        if (np.array(indexes).shape == np.array(indexes2).shape) and (np.array(indexes) == np.array(indexes2)).all():
            if num % 100 == 0:
                print("无法收敛，返回")
                return indexes
            break
        else:
            num = num + 1
            indexes2 = copy.deepcopy(indexes)

    return indexes

# 首先加载模型，根据加载的模型，提取参数
def load_clients(n_clients):
    model_states = [[] for i in range(n_clients)]      # 存放n_clients个模型参数
    for i in range(n_clients):
        grad_model = torch.load('./cache/grad_model_{}.pt'.format(i))       # 获得梯度模型
        # init_para = torch.nn.utils.parameters_to_vector(grad_model.parameters())
        key = ["fc1.weight","fc1.bias","fc2.weight","fc2.bias"
                ,"hidden1.weight","hidden1.bias","hidden2.weight","hidden1.bias","out.weight","out.bias"
                ,"out.weight","out.bias"
                ,"linear.weight","linear.bias"]
        # if i == 0:
        #     print(grad_model.keys())
        for name in grad_model.keys():
            # if name not in key:
            #     continue
            a = grad_model[name].view(-1).tolist()
            model_states[i].extend(a)
            
    return model_states

def Distance_matrix(n_clients,clients):
    distance = []  # 计算距离矩阵
    for i in range(n_clients):
        dis = []
        for j in range(n_clients):
            d = torch.norm(torch.tensor(clients[i]) - torch.tensor(clients[j]), p=2, dim=0).item()  # 计算两个模型之间的欧氏距离
            if i == j:
                d = 1000
            dis.append(d)
        dis[i] = min(dis)
        for c in range(n_clients):
            dis[c] = (max(dis) - dis[c]) / (max(dis) - min(dis))  # 归一化到0-1
        distance.append(dis)
    return distance

def SVD_Dis(n_clients,clients,epoch):


    a = torch.transpose(torch.tensor(clients), 0, 1)  # 矩阵转置
    u, s, v = torch.svd_lowrank(a, q=n_clients)  # 主成分分析，使用奇异值分解

    print(u.shape)
    print(s)
    print(v.shape)
    plt.pie(s, autopct='%1.1f%%')
    plt.savefig("./result/picture/a_{}.jpg".format(epoch))
    plt.show()
    k = 0
    for k in range(len(s)):  # 选取主成分的90%的前k项
        if torch.sum(s[0:k]) / torch.sum(s) >= 0.8:
            break
    print(k)  # 打印簇个数

    return k

