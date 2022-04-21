def param(dataset, get_data_way):
    gamma, lr = 0, 0                            # gamma为退火系数，lr学习率
    mid_epoch, epoch = 0,0                      # 簇模型和全局模型分开训练发生轮次

    if get_data_way == "practical_nonIID":          # 第三种划分数据方式
        gamma = 1

        if dataset == "MNIST" or dataset == "FMNIST":
            lr = 0.003
            epoch = 200
            mid_epoch = 100
        elif dataset == "EMNIST":
            lr = 0.004
            epoch = 200
            mid_epoch = 150
        elif dataset == "CIFAR10":
            lr = 0.005
            epoch = 400
            mid_epoch = 200
        else:
            lr = 0.01
            epoch = 200
            mid_epoch = 100

    elif get_data_way == "nonIID":                  # 每个用户两个标签
        if dataset == "MNIST":
            lr = 0.003
            epoch = 400
            mid_epoch = 20
            gamma = 0.99
        elif dataset == "FMNIST":
            lr = 0.003
            epoch = 300
            mid_epoch = 20
            gamma = 0.99
        elif dataset == "EMNIST":
            lr = 0.02
            epoch = 200
            mid_epoch = 50
            gamma = 0.99
        elif dataset == "CIFAR10":
            lr = 0.02
            epoch = 400
            mid_epoch = 100
            gamma = 0.99
        else:
            lr = 0.02
            epoch = 400
            mid_epoch = 100
            gamma = 0.99

    else:
        gamma = 0.985
        if dataset == "MNIST" or dataset == "FMNIST":
            lr = 0.003
            epoch = 100
            mid_epoch = 80
        elif dataset == "EMNIST":
            lr = 0.004
            epoch = 100
            mid_epoch = 80
        elif dataset == "CIFAR10":
            lr = 0.05
            epoch = 200
            mid_epoch = 180
        else:
            lr = 0.05
            epoch = 200
            mid_epoch = 180

    return mid_epoch, epoch, lr, gamma