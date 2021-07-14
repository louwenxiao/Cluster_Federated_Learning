import numpy as np
import random
import torch
import copy

# 三种聚类算法都算一下


# 这个文件的主要功能是划分簇，返回一个包含k个元素的数组，k是输入
# 其中每个元素表示这个簇内具有的用户
# 输入两个元素，第一个表示用户数量，第二个表示聚类簇的个数
def K_means_cluster(n_clients,k_means):
    clients = load_clients(n_clients=n_clients)
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
        for i in range(k_means):
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


def Hierarchical_clustering(n_clients,k_means):       # 层次聚类
    clients = load_clients(n_clients=n_clients)

    C = []
    for j in range(n_clients):
        C.append([[j],[clients[j]]])

    while len(C)>k_means:
        # 找出距离最近的两个聚类簇，合并
        M = [[0 for i in range(len(C))] for i in range(len(C))]
        for i in range(len(C)):
            for j in range(i+1,len(C)):
                M[i][j] = distance(C[i],C[j])
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


def Density_clustering():            # 密度聚类
    A = 0

# 首先加载模型，根据加载的模型，提取参数
def load_clients(n_clients):
    model_states = [[] for i in range(n_clients)]      # 存放n_clients个模型参数
    for i in range(n_clients):
        model = torch.load('./cache/model_state_{}.pt'.format(i))       # 是字典
        for name in model:
            a = model[name].view(-1).tolist()
            model_states[i].extend(a)
    return model_states


# 计算两个簇之间的距离 [[j],[clients[j]]]
def distance(A,B):
    l = len(A[1][0])
    a = [ 0 for i in range(l)]
    b = [ 0 for i in range(l)]
    for i in A[1]:
        a = np.array(i)/len(A[1]) + a
    for i in B[1]:
        b = np.array(i)/len(B[1]) + b

    dis = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dis
