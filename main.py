from data_and_model.models import get_model         # 调用自己定义的模块
from data_and_model.datasets import download_data
from data_and_model.parameter import param
from clients_and_server.server import get_server
from clients_and_server.clients import get_client
from clients_and_server.cluster import Kmeans, Kmeans_plusplus
from plot import plot_acc,plot_loss

# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

import copy
import torch
import argparse
import time
import sys
import os
import psutil


def main(para):
    args = copy.deepcopy(para)

    print('Download and Initialize Dataset...')
    data_loader = download_data(args=args)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    model = get_model(dataset=args.dataset,model_name=args.model_name)

    clients = []
    for i in range(args.global_nums):
        data = data_loader.get_data()
        user = get_client(id=i,model=copy.deepcopy(model),device=device,dataset=data,args=args)
        clients.append(user)
        # print("Client:{} initialization is complete".format(i))

    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    info = psutil.virtual_memory()
    print(info)


    # create server
    server = get_server(model=copy.deepcopy(model),device=device,clients=clients,args=args)

    print("预训练开始......")
    for client in clients:
        client.pre_train()     # 模型预训练，保存模型梯度到本地
    print("预训练结束......")
    clients_id = Kmeans_plusplus(n_clients=args.global_nums,device=device,epoch='N') # 使用kmeans++进行聚类
    print(clients_id)
    # sys.exit()

    losses = []             # 存放精度和损失
    accuracyes = []
    for epoch in range(args.mid_epoch):
        print("\nepoch：{}".format(epoch+1))

        t1 = time.time()
        for client in clients:  
            client.local_train()        # 用户进行本地训练
        # if epoch%20 == 0:
        #     _ = Kmeans_plusplus(n_clients=args.global_nums,device=device,epoch=epoch)
        #     print("sleep 10s ....")
        #     time.sleep(10)
        t2 = time.time()
        print("用户训练：",t2-t1)

        server.get_cluster_model(clients_id)    # 簇内模型聚合
        t3 = time.time()
        print("簇聚合：",t3-t2)

        server.get_global_model()
        for client in clients:
            client.get_global_model()
        t4 = time.time()
        print("聚合全局模型：",t4-t3)

        accuracyes.append(0)
        losses.append(0)
        loss, acc = 0,0
        for client_num in range(args.global_nums):     # 测试簇模型
            loss,acc = clients[client_num].test_model()
            accuracyes[epoch] = accuracyes[epoch] + acc
            losses[epoch] = losses[epoch] + loss
        accuracyes[epoch] = accuracyes[epoch] / args.global_nums
        t5 = time.time()

        # writer.add_scalar('acc', acc, epoch)
        # writer.add_scalar('loss',loss, epoch)
        

        print("测试时间：",t5-t4)

        print("精度：",accuracyes[epoch])
        print("损失：",losses[epoch])
    # sys.exit()

    # 簇内部单独计算，不再计算全局模型
    for epoch in range(args.mid_epoch,args.epoch):
        print("\nepoch：{}".format(epoch+1))

        for client in clients:  
            client.local_train()        # 用户进行本地训练

        server.get_cluster_model(clients_id)    # 簇内模型聚合

        for client in clients:
            client.get_cluster_model(clients_id)

        # server.client_get_model(clients_id)

        accuracyes.append(0)
        losses.append(0)
        loss, acc = 0, 0
        for cli in range(args.global_nums):     # 测试簇模型
            loss,acc = clients[cli].test_model()
            accuracyes[epoch] = accuracyes[epoch] + acc
            losses[epoch] = losses[epoch] + loss
        accuracyes[epoch] = accuracyes[epoch] / args.global_nums

        print("精度：",accuracyes[epoch])
        print("损失：",losses[epoch])
        # writer.add_scalar('acc' + str(args.idx), acc, epoch)
        # writer.add_scalar('loss' + str(args.idx),loss, epoch)

    plot_acc(accuracyes,args.get_data,args.dataset)
    plot_loss(losses, args.get_data, args.dataset)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST","FMNIST"])
    parser.add_argument("--get_data", type=str, default="practical_nonIID", choices=["IID","nonIID","practical_nonIID"])
    parser.add_argument("--model_name", type=str, default="CNN", choices=["CNN","MCLR","RNN","ResNet18"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01,help="Local learning rate" )
    parser.add_argument("--gamma", type=float, default=0.98, choices=[1, 0.98, 0.99])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epoch", type=int, default=200,help="global train epoch")
    parser.add_argument("--mid_epoch", type=int, default=100,help="change way")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--global_nums", type=int, default=30, help="Number of all Users")
    parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to run,-1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--k", type=int, default=3, help="Number of all clusters")
    parser.add_argument("--pre_epochs", type=int, default=1, help="number of pre epochs")
    args = parser.parse_args()
    
    
    for x in ["MNIST"]:
        way = ["IID"]
            
        for y in way:
            args.dataset,args.get_data = x,y
            args.mid_epoch,args.epoch,args.lr,args.gamma = param(x,y)

            print("=" * 80)  
            print("Summary of training process:")
            print("Dataset: {}".format(args.dataset))        # default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST"]
            print("Get data way: {}".format(args.get_data))  # default="IID", choices=["IID","nonIID","practical_nonIID"]
            print("Model_name: {}".format(args.model_name))
            print("Batch size: {}".format(args.batch_size))  # default=20
            print("Learing rate: {}".format(args.lr))  # default=0.01, help="Local learning rate"
            print("gamma: {}".format(args.gamma))
            print("Momentum: {}".format(args.momentum))
            print("epoch: {}".format(args.epoch))
            print("mid_epoch: {}".format(args.mid_epoch))
            print("Number of local rounds: {}".format(args.local_epochs))
            print("Optimizer: {}".format(args.optimizer))  # default="SGD"
            print("All users: {}".format(args.global_nums))     # default=100, help="Number of all Users"
            print("=" * 80)
            
            main(args)


