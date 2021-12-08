import torch
import time
import random
from torch.serialization import validate_cuda_device
random.seed(int(time.time())%100000)
from torch.utils.data import DataLoader, dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
import sys
import os
import psutil


class download_data(object):
    def __init__(self, args):
        self.args = args
        self.data_name = args.dataset
        self.batch_size = args.batch_size
        self.get_data_way = args.get_data
        self.data_num = [[] for _ in range(args.k)]         # 用来记录第三种数据，每个用户所在组
        self.client_num = 0
        self.__load_dataset()
        self.__initial()

    # initial self.train_dataset and self.test_dataset
    def __load_dataset(self,path = "/data/wxlou/dataset"):
        # dataset path
        if self.data_name == 'MNIST':
            train_dataset = datasets.MNIST(path,
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

            test_dataset = datasets.MNIST(path,
                                          train=False,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

        elif self.data_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10(path,
                                             train=True,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.RandomCrop(32, 4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010))
                                             ]))
            test_dataset = datasets.CIFAR10(path,
                                            train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010))
                                            ]))

        elif self.data_name == "FMNIST":
            train_dataset = datasets.FashionMNIST(path,
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                  ]))
            test_dataset = datasets.FashionMNIST(path,
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))

        elif self.data_name == 'EMNIST':
            train_dataset = datasets.EMNIST(path,
                                            split="byclass",
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
            test_dataset = datasets.EMNIST(path,
                                           split="byclass",
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

        elif self.data_name == 'CIFAR100':
            train_dataset = datasets.CIFAR100(path,
                                              train = True,
                                              download = True,
                                              transform=transforms.Compose([
                                                    transforms.RandomCrop(32, 4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                      (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                                                ]))
            test_dataset = datasets.CIFAR100(path,
                                             train = False,
                                             download = True,
                                             transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                                             ]))
        else:
            raise RuntimeError('the name inputed is wrong!')

        # self.train_dataset = list(train_dataset)
        # self.test_dataset = list(test_dataset)
        # print(self.train_dataset[0])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # print(self.train_dataset[0])
        # sys.exit()

    # 初始化
    def __initial(self):
        if self.get_data_way == "IID":  # IID, datasets are divided evenly
            num = int(len(self.train_dataset) / self.args.global_nums)
            train_split = [num for _ in range(self.args.global_nums)]
            train_split.append(len(self.train_dataset) - self.args.global_nums * num)

            num = int(len(self.test_dataset) / self.args.global_nums)
            test_split = [num for i in range(self.args.global_nums)]
            test_split.append(len(self.test_dataset) - self.args.global_nums * num)

            train_dataset = random_split(self.train_dataset, train_split)
            test_dataset = random_split(self.test_dataset, test_split)
            train_dataset = [DataLoader(train_dataset[i], batch_size=self.batch_size, shuffle=True,
                                        pin_memory=True) for i in range(self.args.global_nums)]
            test_dataset = [DataLoader(test_dataset[i], batch_size=self.batch_size, shuffle=True,
                                       pin_memory=True) for i in range(self.args.global_nums)]

            self.train_dataset = iter(train_dataset)
            self.test_dataset = iter(test_dataset)

        elif self.get_data_way == "nonIID":         # two classes per client
            self.ranked = []
            if self.data_name == 'EMNIST':
                self.index_train = [[] for i in range(0, 62)]
                self.index_test = [[] for i in range(0, 62)]

            elif self.data_name == 'CIFAR100':
                self.index_train = [[] for i in range(0, 100)]
                self.index_test = [[] for i in range(0, 100)]

            else:
                self.index_train = [[] for i in range(0, 10)]
                self.index_test = [[] for i in range(0, 10)]

            for i, data in enumerate(self.train_dataset):       # sort by label
                self.index_train[data[1]].append(i)
            for i, data in enumerate(self.test_dataset):
                self.index_test[data[1]].append(i)

            # self.indexes = []

        else:
            if self.args.k == 3:
                self.index_train = [[], [], []]     # divide the datasets into three disjoint sets
                self.index_test = [[], [], []]
                if self.data_name == 'EMNIST':
                    for i, data in enumerate(self.train_dataset):
                        if data[1] < 10:
                            self.index_train[0].append(i)
                        elif data[1] > 35:
                            self.index_train[2].append(i)
                        else:
                            self.index_train[1].append(i)
                    for i, data in enumerate(self.test_dataset):
                        if data[1] < 10:
                            self.index_test[0].append(i)
                        elif data[1] > 35:
                            self.index_test[2].append(i)
                        else:
                            self.index_test[1].append(i)
                elif self.data_name == 'CIFAR100':
                    for i, data in enumerate(self.train_dataset):
                        if data[1] < 33:
                            self.index_train[0].append(i)
                        elif data[1] > 66:
                            self.index_train[2].append(i)
                        else:
                            self.index_train[1].append(i)
                    for i, data in enumerate(self.test_dataset):
                        if data[1] < 33:
                            self.index_test[0].append(i)
                        elif data[1] > 66:
                            self.index_test[2].append(i)
                        else:
                            self.index_test[1].append(i)
                else:
                    for i, data in enumerate(self.train_dataset):
                        if data[1] < 3:
                            self.index_train[0].append(i)
                        elif data[1] > 5:
                            self.index_train[2].append(i)
                        else:
                            self.index_train[1].append(i)
                    for i, data in enumerate(self.test_dataset):
                        if data[1] < 3:
                            self.index_test[0].append(i)
                        elif data[1] > 5:
                            self.index_test[2].append(i)
                        else:
                            self.index_test[1].append(i)

            else:
                self.index_train = [[], [], [], [], []]  # divide the datasets into five disjoint sets
                self.index_test = [[], [], [], [], []]
                if self.data_name == 'EMNIST':
                    for i, data in enumerate(self.train_dataset):
                        if data[1] < 12:
                            self.index_train[0].append(i)
                        elif data[1] < 24:
                            self.index_train[1].append(i)
                        elif data[1] < 36:
                            self.index_train[2].append(i)
                        elif data[1] < 48:
                            self.index_train[3].append(i)
                        else:
                            self.index_train[4].append(i)
                    for i, data in enumerate(self.test_dataset):
                        if data[1] < 12:
                            self.index_test[0].append(i)
                        elif data[1] < 24:
                            self.index_test[1].append(i)
                        elif data[1] < 36:
                            self.index_test[2].append(i)
                        elif data[1] < 48:
                            self.index_test[3].append(i)
                        else:
                            self.index_test[4].append(i)
                elif self.data_name == 'CIFAR100':
                    for i, data in enumerate(self.train_dataset):
                        if data[1] < 20:
                            self.index_train[0].append(i)
                        elif data[1] < 40:
                            self.index_train[1].append(i)
                        elif data[1] < 60:
                            self.index_train[2].append(i)
                        elif data[1] < 80:
                            self.index_train[3].append(i)
                        else:
                            self.index_train[4].append(i)
                    for i, data in enumerate(self.test_dataset):
                        if data[1] < 20:
                            self.index_test[0].append(i)
                        elif data[1] < 40:
                            self.index_test[1].append(i)
                        elif data[1] < 60:
                            self.index_test[2].append(i)
                        elif data[1] < 80:
                            self.index_test[3].append(i)
                        else:
                            self.index_test[4].append(i)
                else:
                    for i, data in enumerate(self.train_dataset):
                        if data[1] < 2:
                            self.index_train[0].append(i)
                        elif data[1] < 4:
                            self.index_train[1].append(i)
                        elif data[1] < 6:
                            self.index_train[2].append(i)
                        elif data[1] < 8:
                            self.index_train[3].append(i)
                        else:
                            self.index_train[4].append(i)
                    for i, data in enumerate(self.test_dataset):
                        if data[1] < 2:
                            self.index_test[0].append(i)
                        elif data[1] < 4:
                            self.index_test[1].append(i)
                        elif data[1] < 6:
                            self.index_test[2].append(i)
                        elif data[1] < 8:
                            self.index_test[3].append(i)
                        else:
                            self.index_test[4].append(i)


    # get IID datasets
    def get_IID_data(self):
        print("get_IID_data")

        # train_sample = random.sample(range(len(self.train_dataset)),train_num)
        # train_data = torch.utils.data.Subset(self.train_dataset,train_sample)
        # train_dataset = DataLoader(train_data,batch_size=self.batch_size, shuffle=True)
        #
        # test_sample = random.sample(range(len(self.test_dataset)),test_num)
        # test_data = torch.utils.data.Subset(self.test_dataset,test_sample)
        # test_dataset= DataLoader(test_data,batch_size=self.batch_size, shuffle=True)

        train_data = next(self.train_dataset)
        test_data = next(self.test_dataset)
        data = []
        data.append(train_data)
        data.append(test_data)
        return data


    def get_nonIID_data(self):
        print("get_nonIID_data")

        if self.args.dataset == "EMNSIT":
            rank = random.sample(range(62),2)
        elif self.args.dataset == "CIFAR100":
            rank = random.sample(range(100), 2)
        else:
            rank = random.sample(range(10),2)

        print(rank)

        train_index = []
        test_index = []
        for i in rank:
            train_index.extend(self.index_train[i])
            test_index.extend(self.index_test[i])

        # 下面的代码表示抽样
        # train_index = random.sample(train_index, int(len(train_index) * rata))
        # test_index = random.sample(test_index, int(len(test_index) * rata))
        sample_train = train_index
        sample_test = test_index

        dataset1 = torch.utils.data.Subset(self.train_dataset, sample_train)
        dataset2 = torch.utils.data.Subset(self.test_dataset, sample_test)
        train_dataset = DataLoader(dataset1, batch_size=self.batch_size, shuffle=True)
        test_dataset = DataLoader(dataset2, batch_size=self.batch_size, shuffle=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data


    # 获得practical_nonIID_data，这个更加符合实际
    # 将所有的数据分为三个组，每个组内的数据不完全一样。
    # 每个组具有80%的主要数据，20%的其他数据
    def get_practical_nonIID_data(self,id=None):
        print("get_practical_nonIID_data")
        # id表示更新时用户的编号，id=None时用来记录最初随机分配
        if id == None:
            n = 3
        else:
            n = 5

        data_index = random.randint(0,n-1)        # Gets the specified index dataset
        self.update_data(id=id,data=data_index)   # update

        print(data_index)
        if self.data_name == "EMNIST":
            train_data_num = 10000          # train_dataset_num
        else:
            train_data_num = 2000

        test_data_num = int(train_data_num * 0.2)       # 测试集大小

        sampling_train = random.sample(self.index_train[data_index], int(train_data_num*0.8))    # Obtain major component train_data
        other_train = []
        for i in range(n):
            if i != data_index:
                other_train.extend(self.index_train[i])
        other_train = random.sample(other_train, int(train_data_num * 0.2))             # Obtain non-major component train_data
        sampling_train.extend(other_train)
        train_dataset = torch.utils.data.Subset(self.train_dataset, sampling_train)
        train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        sampling_test = random.sample(self.index_test[data_index], int(test_data_num*0.8))
        other_test = []
        for i in range(n):
            if i != data_index:
                other_test.extend(self.index_test[i])
        other_test = random.sample(other_test, int(test_data_num * 0.2))
        sampling_test.extend(other_test)
        test_dataset = torch.utils.data.Subset(self.test_dataset, sampling_test)
        test_dataset = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data

    # update
    def update_data(self,id=None,data=None):
        if data == None:
            print("数据产生错误...")
            sys.exit()
        
        if id == None:      # 表示初始化数据
            self.data_num[data].append(self.client_num)
            self.client_num += 1
        else:
            for data_index in self.data_num:    # 删除之前id已经存在位置
                if id in data_index:
                    data_index.remove(id)
            self.data_num[data].append(id)   # 添加新成员并排序
            self.data_num[data].sort()
        for data_index in self.data_num:print(data_index)   # 用来查看
            

    def get_data(self, get_data_way="train",id = None):
        if get_data_way == "test":
            data = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
            return data

        if self.get_data_way == "IID":
            data = self.get_IID_data()
        elif self.get_data_way == "nonIID":
            data = self.get_nonIID_data()
        else:
            data = self.get_practical_nonIID_data(id)

        return data
