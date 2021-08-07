import torch
import random
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
import sys
import time
import os
import psutil


# 获得数据，默认是MNIST数据集，同时还有CIFAR、EMNIST、FMNIST数据集
# 对于non-IID数据：每个用户随机获得两个标签
class download_data(object):
    def __init__(self, args):
        self.args = args
        self.data_name = args.dataset
        self.batch_size = args.batch_size
        self.get_data_way = args.get_data
        self.__load_dataset()         # 初始化直接调用函数
        self.__initial()


    # 产生 self.train_dataset 和 self.test_dataset
    def __load_dataset(self):
        if self.data_name =='MNIST':
            train_dataset = datasets.MNIST('./data/',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]))
            test_dataset = datasets.MNIST('./data/',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))

        elif self.data_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10('./data/',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                           (0.2023, 0.1994, 0.2010))
                                                  ]))
            test_dataset = datasets.CIFAR10('./data/',
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010))
                                                 ]))

        elif self.data_name == "FMNIST":
            train_dataset = datasets.FashionMNIST('./data/',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]))
            test_dataset = datasets.FashionMNIST('./data/',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))

        elif self.data_name == 'EMNIST':
            train_dataset = datasets.EMNIST('./data/',
                                            split="byclass",
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
            test_dataset = datasets.EMNIST('./data/',
                                            split="byclass",
                                            train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))

        else:
            raise RuntimeError('the name inputed is wrong!')

        self.train_dataset = list(train_dataset)
        self.test_dataset = list(test_dataset)
        # self.train_dataset = train_dataset
        # self.test_dataset = test_dataset

    # 初始化
    def __initial(self):
        if self.get_data_way == "IID":      # 获得IID数据：用户平分数据集
            num = int(len(self.train_dataset)/self.args.global_nums)
            train_split = [num for i in range(self.args.global_nums)]
            train_split.append(len(self.train_dataset)-self.args.global_nums*num)
            
            num = int(len(self.test_dataset)/self.args.global_nums)
            test_split = [num for i in range(self.args.global_nums)]
            test_split.append(len(self.test_dataset)-self.args.global_nums*num)

            train_dataset = random_split(self.train_dataset,train_split)
            test_dataset = random_split(self.test_dataset,test_split)
            train_dataset = [DataLoader(train_dataset[i], batch_size=self.batch_size, shuffle=True,
                                        pin_memory=True) for i in range(self.args.global_nums)]
            test_dataset = [DataLoader(test_dataset[i], batch_size=self.batch_size, shuffle=True,
                                        pin_memory=True) for i in range(self.args.global_nums)]
            
            self.data = list(zip(train_dataset,test_dataset))
            self.num = -1                # 记录所在位置
            print(sys.getsizeof(self.train_dataset[0][0]),sys.getsizeof(self.train_dataset[0][1]))

        elif self.get_data_way == "nonIID":
            if self.data_name == 'EMNIST':
                self.index_train = [[] for i in range(0,62)]
                self.index_test = [[] for i in range(0,62)]
            else:
                self.index_train = [[] for i in range(0,10)]
                self.index_test = [[] for i in range(0,10)]

            for i, data in enumerate(self.train_dataset):
                self.index_train[data[1]].append(i)           # 按照标签分类
            for i, data in enumerate(self.test_dataset):
                self.index_test[data[1]].append(i)           # 按照标签分类

            self.indexes = []

        else:
            self.index_train = [[],[],[]]       # 将数据集分成3个不相交的集合
            self.index_test = [[],[],[]]

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

    # 获得IID数据，用户平分数据集
    # 每个用户具有相同的样本数量，相同的标签
    def get_IID_data(self):   # 设置训练样本比例

        self.num = self.num + 1
        return self.data[self.num]


    # 获得non-IID数据，每个用户仅仅包含两个标签,需要考虑三种数据集的不同
    # 一个是0-10，另一个是10个字符串，还有一个是62个标签
    def get_nonIID_data(self):
        if self.data_name == "EMNIST":
            num = 62
        else:
            num = 10
        rank = random.sample(range(0,num),2)
        indexes = [i for i in range(num) if i not in self.indexes]      # 不存在标签
        if len(indexes) != 0:
            if rank[0] not in indexes:
                rank[1] = indexes[0]
        print(rank)

        for i in rank:
            if i not in self.indexes:
                self.indexes.append(i)
        self.indexes.sort()
        print(self.indexes)
        time.sleep(2)

        train_index = []
        test_index = []
        for i in rank:
            train_index.extend(self.index_train[i])
            test_index.extend(self.index_test[i])

        sample_train = random.sample(train_index,int(len(train_index)*0.5))
        sample_test = random.sample(test_index,int(len(test_index)*0.5))

        dataset1 = torch.utils.data.Subset(self.train_dataset, sample_train)    # 获取指定元素的数据集
        dataset2 = torch.utils.data.Subset(self.test_dataset, sample_test)    # 获取指定元素的数据集
        train_dataset = DataLoader(dataset1, batch_size=self.batch_size, shuffle=True)
        test_dataset = DataLoader(dataset2, batch_size=self.batch_size, shuffle=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data


    # 获得practical_nonIID_data，这个更加符合实际
    # 将所有的数据分为三个组，每个组内的数据不完全一样。
    # 每个组具有90%的主要数据，10%的其他数据
    # data_num表示选用那个数据集作为主要数据
    def get_practical_nonIID_data(self,data_num=0):
        print(data_num)
        train_data_num = random.randint(5000,20000)      # 训练集大小
        sampling_train = random.sample(self.index_train[data_num],train_data_num)
        other1 = []
        for i in range(3):
            if i != data_num:
                other1.extend(self.index_train[i])
        other1 = random.sample(other1,int(train_data_num*0.1))
        sampling_train.extend(other1)

        dataset = torch.utils.data.Subset(self.train_dataset, sampling_train)
        train_dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        test_data_num = int(train_data_num*0.2)      # 测试集大小
        sampling_test = random.sample(self.index_test[data_num],test_data_num)
        other1 = []
        for i in range(3):
            if i != data_num:
                other1.extend(self.index_test[i])
        other1 = random.sample(other1,int(test_data_num*0.1))
        sampling_test.extend(other1)

        test_dataset = torch.utils.data.Subset(self.test_dataset, sampling_test)
        test_dataset = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        
        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data


    # 第一种方式：IID；第二种获得数据的方式：每个用户随机获得2个标签，成为nonIID；
    # 第三种获得数据的方式，将数据划分为三组，每一组的用户数据相似，组之间数据相似小
    def get_data(self,data_num=0,get_data="train"):
        if get_data == "test":      # 如果仅仅得到测试数据，这样获得
            data = DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=True)
            return data

        if self.get_data_way == "IID":
            data = self.get_IID_data()
        elif self.get_data_way == "nonIID":
            data = self.get_nonIID_data()
        else:  # 划分三组，每一组内的数据相似，组间不相似
            data = self.get_practical_nonIID_data(data_num)

        return data
