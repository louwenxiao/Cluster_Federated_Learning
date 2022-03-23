from data_and_model.models import get_model
from data_and_model.datasets import download_data
from data_and_model.parameter import param
from clients_and_server.server import server
from clients_and_server.clients import client
from plot import plot_acc,plot_loss
import copy
import torch
import argparse
import time
import os
import psutil

def main(a):
    
    args = copy.deepcopy(a)
    print('Initialize Dataset...')
    data_loader = download_data(args=args)

    test_data = data_loader.get_data(get_data_way="test")

    # 第一种方式：IID；第二种获得数据的方式：每个用户随机获得2个标签，称为nonIID；
    # 第三种获得数据的方式，将数据划分为三组，每一组的用户数据相似，组之间数据非常不相似
    m = get_model(dataset=args.dataset,model_name=args.model_name)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    # 获得云模型，获得global_nums个用户，放在clients中
    clients = []
    for i in range(args.global_nums):

        mid_user = client(id=i,model=copy.deepcopy(m),device=device,dataset=data_loader.get_data(),args=args)
        clients.append(mid_user)
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
        print("---------------{}---------------".format(i))


    losses = []
    accuracyes = []
    s = server(model=copy.deepcopy(m),device=device, dataset=test_data,clients = clients,args = args)

    s.Group_Cold_Start(k=3)                # 组冷启动

    for epoch in range(args.epoch):
        losses.append(0)
        accuracyes.append(0)

        print("\n第{}轮：".format(epoch))
        for c in clients:
            c.local_train()            # 组内训练
        s.IntraGroupUpdate()                # 组内用户并行训练，并且内部更新

        s.InterGroupAggregation()           # 组间更新

        s.Clients_get_model()

        for c in clients:
            loss,acc = c.test_model()
            losses[epoch] += loss
            accuracyes[epoch] += acc

        losses[epoch] = losses[epoch]/len(clients)
        accuracyes[epoch] = accuracyes[epoch]/len(clients)
        print("精度：",accuracyes[epoch])
        print("损失：",losses[epoch])

    plot_acc(accuracyes,args.get_data,args.dataset)
    plot_loss(losses, args.get_data, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST","FMNIST"])
    parser.add_argument("--get_data", type=str, default="IID", choices=["IID","nonIID","practical_nonIID"])
    parser.add_argument("--model_name", type=str, default="CNN", choices=["CNN","MCLR","RNN","ResNet18"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01,help="Local learning rate" )
    parser.add_argument("--gamma", type=float, default=0.98, choices=[1, 0.98, 0.99])
    parser.add_argument("--momentum", type=float, default=0.9, choices=[0.9,0.5])
    parser.add_argument("--epoch", type=int, default=200,help="global train epoch")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--global_nums", type=int, default=50, help="Number of all Users")
    parser.add_argument("--k", type=int, default=3, help="Number of all clusters")
    parser.add_argument("--gpu", type=int, default=2, help="Which GPU to run,-1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()
    
    
    for x in ["CIFAR100"]:
        way = ["IID"]
            
        for y in way:
            args.dataset,args.get_data = x,y
            _,args.epoch,args.lr,args.gamma = param(x,y)
            args.gamma = 0.99
            if x == "CIFAR10":
                args.epoch = 300

            print("=" * 80)  
            print("Summary of training process:")
            print("Dataset: {}".format(args.dataset))        # default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST"]
            print("Get data way: {}".format(args.get_data))  # default="IID", choices=["IID","nonIID","practical_nonIID"]
            print("Batch size: {}".format(args.batch_size))  # default=20
            print("Learing rate: {}".format(args.lr))  # default=0.01, help="Local learning rate"
            print("gamma: {}".format(args.gamma))
            print("Number of global rounds: {}".format(args.epoch))    # default=800
            print("Momentum: {}".format(args.momentum))
            print("Number of local rounds: {}".format(args.local_epochs))         # default=30
            print("Optimizer: {}".format(args.optimizer))                         # default="SGD"
            print("All users: {}".format(args.global_nums))     # default=100, help="Number of all Users"
            print("=" * 80)
            
            main(args)
