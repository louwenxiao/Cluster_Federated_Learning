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
    a = torch.transpose(torch.tensor(clients),0,1)     
    u,s,v = torch.svd_lowrank(a,q=50) 
    
    print(u.shape)
    print(s.shape)
    print(v.shape)
    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()


    initial = random.sample(range(0, n_clients), k_means)  
    init_model = [clients[i] for i in initial]
    indexes2 = [[] for i in range(k_means)] 
    print("init_model:",initial)
    print("shape:",np.array(clients).shape)
    num = 1

    while True:
        clusters = [[] for i in range(k_means)]   
        indexes = [[] for i in range(k_means)]      

        for i in range(n_clients):
      
            distance = []
            for j in range(k_means):
                a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(init_model[j]))))
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)


        print(np.array(clusters).shape)
        for i in range(k_means):
            print(i)
            a = np.array(clusters[i][0])*0.0
            for j in clusters[i]:
                a = np.array(a) + np.array(j)
            a = np.array(a) / len(clusters[i])
            init_model[i] = a.tolist()

        print(num, indexes, indexes2,'\n')
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
    u,s,v = torch.svd_lowrank(torch.tensor(clients),q=100)            
    
    print(u.shape)
    print(s)
    print(v.shape)
    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()

    for k in range(len(s)):             
        if torch.sum(s[0:k])/torch.sum(s) >= 0.9:
            break
    print(k)               

    result = KMeans(k,max_iter=100).fit(clients).labels_    # kmeans++
    print(result)
    cluster = [[] for _ in range(k)]
    for i,index in enumerate(result):
        cluster[index].append(i)

    return cluster


def Kmeans_plusplus(n_clients,device,epoch):
    clients = load_clients(n_clients=n_clients)
    clients = torch.tensor(clients).to(device)

    distance = Distance_matrix(n_clients, clients)      
    clients = torch.tensor(distance)

    k = SVD_Dis(n_clients, clients, epoch)

    initial_model = [clients[0]]     
    # print(initial_model[0])
    while len(initial_model) < k:
        distances = []              
        for c in clients:
            min_dis = 100000       
            for i in initial_model:
                dis = torch.norm(i-c,p=2,dim=0).item()
                if dis < min_dis:
                    min_dis = dis  
            distances.append(min_dis)
        # print(distances)
        max_index = distances.index(max(distances))  
        distances[max_index] = 0       

        max_index = distances.index(max(distances))  
        initial_model.append(clients[max_index])

    num = 1
    indexes2 = [[] for i in range(k)]  
    while True:
        clusters = [[] for _ in range(k)]    
        indexes = [[] for _ in range(k)]     

        for i in range(n_clients):
            
            distance = []
            for j in range(k):
                # a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(initial_model[j]))))
                a = torch.norm(clients[i]-initial_model[j],p=2,dim=0).item()
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)

        
        for i in range(k):
            a = clusters[i][0]*0.0
            for j in clusters[i]:
                a += j
            a = torch.div(a,len(clusters[i]))
            initial_model[i] = a
            
        if (np.array(indexes).shape == np.array(indexes2).shape) and (np.array(indexes) == np.array(indexes2)).all():
            if num % 100 == 0:
                print("无法收敛，返回")
                return indexes
            break
        else:
            num = num + 1
            indexes2 = copy.deepcopy(indexes)

    return indexes

# load model and get para
def load_clients(n_clients):
    model_states = [[] for i in range(n_clients)]      
    for i in range(n_clients):
        grad_model = torch.load('./cache/grad_model_{}.pt'.format(i))       
        
        for name in grad_model.keys():
            a = grad_model[name].view(-1).tolist()
            model_states[i].extend(a)
            
    return model_states

# compute distance matrix
def Distance_matrix(n_clients,clients):
    distance = []  
    for i in range(n_clients):
        dis = []
        for j in range(n_clients):
            d = torch.norm(torch.tensor(clients[i]) - torch.tensor(clients[j]), p=2, dim=0).item()  
            if i == j:
                d = 1000
            dis.append(d)
        dis[i] = min(dis)
        for c in range(n_clients):
            dis[c] = (max(dis) - dis[c]) / (max(dis) - min(dis))  
        distance.append(dis)
    return distance


# get cluster number by using SVD
def SVD_Dis(n_clients,clients,epoch):

    a = torch.transpose(torch.tensor(clients), 0, 1)  
    u, s, v = torch.svd_lowrank(a, q=n_clients)  

    print(u.shape)
    print(s)
    print(v.shape)
    plt.pie(s, autopct='%1.1f%%')
    plt.savefig("./result/picture/a_{}.jpg".format(epoch))
    plt.show()
    k = 0
    for k in range(len(s)):  
        if torch.sum(s[0:k]) / torch.sum(s) >= 0.8:
            break
    print(k)  

    return k

