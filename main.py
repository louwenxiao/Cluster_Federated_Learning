from data_and_model.models import get_model
from data_and_model.datasets import download_data
from clients_and_server.server import server
from clients_and_server.clients import client
from clients_and_server.cluster import K_means_cluster
from plot import plot_acc
import copy
import torch
import argparse
import os
# import time
# from torch import nn
# import torch.multiprocessing as mp
# from multiprocessing import Process          # 导入多进程中的进程池
# from torch.autograd import Variable


def main(dataset,get_data_way,model,batch_size,learning_rate,num_glob_iters,
         local_epochs,optimizer,global_nums,gpu,k,pre_epochs):

    print('Initialize Dataset...')
    data_loader = download_data(dataset_name=dataset, batch_size=batch_size)
    # 全局模型的测试用数据，使用全部的测试数据
    test_data = data_loader.get_data(get_data_way='IID')[1]
    # 第一种方式：IID；第二种获得数据的方式：每个用户随机获得2个标签，成为nonIID；
    # 第三种获得数据的方式，将数据划分为三组，每一组的用户数据相似，组之间数据非常不相似

    m = get_model(dataset=dataset,model=model)
    #mp.set_start_method('spawn')
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    # 获得云模型，获得global_nums个用户，放在clients中
    clients = []
    for i in range(global_nums):
        mid_user = client(id=i,model=copy.deepcopy(m),device=device,dataset=data_loader.get_data(get_data_way=get_data_way),
                          learning_rate=learning_rate,optimizer=optimizer,local_epochs=local_epochs,server_id=0)
        clients.append(mid_user)
        clients[i].pre_train(epoch=pre_epochs)
        print("---------------{}---------------".format(i))

    clouds = []
    clients_id = K_means_cluster(n_clients=global_nums,k_means=k)
    print(clients_id)
    
    for i in range(k):
        cloud = server(model=copy.deepcopy(m),device=device, dataset=data_loader.get_data(get_data_way=get_data_way),
                       clients_id=clients_id[i],server_id=i)
        clouds.append(cloud)
        clouds[i].aggregate_model()
        for j in clients_id[i]:         # 更新用户所在簇
            clients[j].updata_clu(id=i)
            clients[j].get_cluster_modal()

    accuracyes = [[],[],[]]
    s = server(model=m,device=device, dataset=data_loader.get_data(get_data_way="IID"),
                       clients_id=range(k),server_id=10)    
    s.aggregate_cluster()
    # server_id=10 没有任何意义，纯粹为了填写一个参数而已
    for epoch in range(num_glob_iters):
        print("\n第{}轮：".format(epoch))
        accuracyes[0].append(0)             # 第一个元素为每个用户的平均精度
        accuracyes[1].append(0)             # 第二个为簇模型本地数据
        accuracyes[2].append(0)             # 第三个全局模型全部数据

        # 将全局模型送给每一个用户。
        for j in range(global_nums):        
            clients[j].get_model()

        # 簇内部聚合,L 表示簇内部训练次数
        L = 1
        for clu in range(k):

            for i in range(L):          
                for cli in clients_id[clu]:         
                    clients[cli].local_train()
                    _,acc = clients[cli].test_model()
                    accuracyes[0][epoch] += acc
                    print("num{}:".format(cli),acc)
                    
                clouds[clu].aggregate_model()   # 簇内部模型聚合

                for cli in clients_id[clu]:     # 将模型送给用户
                    clients[cli].get_cluster_modal()
        accuracyes[0][epoch] = accuracyes[0][epoch]/(L*global_nums)

        for j in range(global_nums):            # 获得这个簇在所有本地用户数据上的精度
            _,acc = clients[j].test_model()
            accuracyes[1][epoch] += acc
        accuracyes[1][epoch] = accuracyes[1][epoch]/global_nums  

        s.aggregate_cluster()                   # 聚合所有簇的模型
        accuracyes[2][epoch] = s.gain_acc()
        print(accuracyes[0][epoch],accuracyes[1][epoch],accuracyes[2][epoch])      

    plot_acc(accuracyes,get_data_way,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST"])
    parser.add_argument("--get_data", type=str, default="practical_nonIID", choices=["IID","nonIID","practical_nonIID"])
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "DNN", "MCLR_Logistic"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--num_global_iters", type=int, default=100,help="global train epoch")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--global_nums", type=int, default=30, help="Number of all Users")
    parser.add_argument("--gpu", type=int, default=1, help="Which GPU to run,-1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--k", type=int, default=3, help="Number of all clusters")
    parser.add_argument("--pre_epochs", type=int, default=1, help="number of pre epochs")
    args = parser.parse_args()

    print("=" * 80)  
    print("Summary of training process:")
    print("Dataset: {}".format(args.dataset))        # default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST"]
    print("Get data way: {}".format(args.get_data))  # default="IID", choices=["IID","nonIID","practical_nonIID"]
    print("Local Model: {}".format(args.model))      # default="CNN", choices=["CNN", "DNN", "MCLR_Logistic"]
    print("Batch size: {}".format(args.batch_size))  # default=20
    print("Learing rate: {}".format(args.learning_rate))  # default=0.01, help="Local learning rate"
    print("Number of global rounds: {}".format(args.num_global_iters))    # default=800
    print("Number of local rounds: {}".format(args.local_epochs))         # default=30
    print("Optimizer: {}".format(args.optimizer))                         # default="SGD"
    print("All users: {}".format(args.global_nums))     # default=100, help="Number of all Users"
    print("=" * 80)
    
    main(dataset=args.dataset,
            get_data_way=args.get_data,
            model=args.model,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_glob_iters=args.num_global_iters,
            local_epochs=args.local_epochs,
            optimizer=args.optimizer,
            global_nums=args.global_nums,
            gpu=args.gpu,
            k=args.k,
            pre_epochs=args.pre_epochs)

    # 清除模型，保留结果
    # for i in range(args.global_nums):
    #     os.remove('./cache/model_state_{}.pt'.format(i))
    # for i in range(args.k):
    #     os.remove('./cache/global_model_{}.pt'.format(i))
    # os.remove('./cache/global_model.pt')
