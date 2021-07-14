import torch
import random
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import struct
import time
import gc

# 获得数据，默认是MNIST数据集，同时还有CIFAR和EMNIST数据集
# 获得IID数据，或者获得non-IID数据
# 对于IID数据
class download_data(object):
    def __init__(self, dataset_name='MNIST', batch_size=64):
        self.data_name = dataset_name
        self.batch_size = batch_size
        self.__load_dataset()         # 初始化直接调用函数

    # 产生 self.train_dataset 和 self.test_dataset
    def __load_dataset(self):
        if self.data_name =='MNIST':
            self.train_dataset = datasets.MNIST('./data/',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]))

            self.test_dataset = datasets.MNIST('./data/',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))

        elif self.data_name == 'CIFAR10':
            self.train_dataset = datasets.CIFAR10('./data/',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                  ]))
            self.test_dataset = datasets.CIFAR10('./data/',
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                 ]))

        elif self.data_name == 'EMNIST':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

            # 获得训练集,并且删除中间文件
            train_dataset = []
            train_data = './data/EMNIST/data/emnist-byclass-train-images-idx3-ubyte'
            train_label = './data/EMNIST/data/emnist-byclass-train-labels-idx1-ubyte'
            with open(train_label, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                train_labels = np.fromfile(lbpath, dtype=np.uint8)
            with open(train_data, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)
            for i, j in zip(train_images, train_labels):
                images = np.reshape(i, [28, 28])
                train_dataset.append((transform(images), j))

            sampling = random.sample(range(0,len(train_dataset)),int(0.2*len(train_dataset)))
            self.train_dataset = torch.utils.data.Subset(train_dataset,sampling)
            del train_images,train_labels,train_dataset,sampling
            gc.collect()

            # 获得测试集,并且删除中间文件
            test_dataset = []
            test_data = './data/EMNIST/data/emnist-byclass-test-images-idx3-ubyte'
            test_label = './data/EMNIST/data/emnist-byclass-test-labels-idx1-ubyte'
            with open(test_label, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                test_labels = np.fromfile(lbpath, dtype=np.uint8)
            with open(test_data, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)
            for i, j in zip(test_images, test_labels):
                images = np.reshape(i, [28, 28])
                test_dataset.append((transform(images), j))

            sampling = random.sample(range(0,len(test_dataset)),int(0.2*len(test_dataset)))
            self.test_dataset = torch.utils.data.Subset(test_dataset,sampling)
            del test_images,test_labels,sampling,test_dataset
            gc.collect()

        else:
            raise RuntimeError('the name inputed is wrong!')
        
        # sampling_train = random.sample(range(0,len(self.train_dataset)),int(0.5*len(self.train_dataset)))
        # self.train_dataset = torch.utils.data.Subset(self.train_dataset, sampling_train)
        # sampling_test = random.sample(range(0,len(self.test_dataset)), int(0.5*len(self.test_dataset)))
        # self.test_dataset = torch.utils.data.Subset(self.test_dataset, sampling_test)


    # 获得IID数据，每个用户样本数量为 train_num=16000,test_num=3200
    # 每个用户具有相同的样本数量，相同的标签
    def get_IID_data(self,train_num=1,test_num=1):   # 设置训练样本比例
        if self.data_name == "EMNIST":
            train_num = 0.1
            test_num = 0.1

        # sampling_train = random.sample(range(0,len(self.train_dataset)),int(train_num*len(self.train_dataset)))
        # train_dataset = torch.utils.data.Subset(self.train_dataset, sampling_train)
        train_dataset = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8,pin_memory=True)

        # sampling_test = random.sample(range(0,len(self.test_dataset)), int(test_num*len(self.test_dataset)))
        # test_dataset = torch.utils.data.Subset(self.test_dataset, sampling_test)
        test_dataset = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8,pin_memory=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data


    # 获得non-IID数据，每个用户仅仅包含两个标签,需要考虑三种数据集的不同
    # 一个是0-10，另一个是10个字符串，还有一个是62个标签
    def get_nonIID_data(self,train_num=1,test_num=1):
        if self.data_name == "EMNIST":
            train_num = 0.2
            test_num = 0.2

        label = []   # 存放选中的标签
        dataset_indices = []
        testdata_indices = []
        if (self.data_name == 'MNIST') or (self.data_name == 'CIFAR10'):
            rank = random.sample(range(0, 10), 2)
            #indices = [[] for i in range(10)]
        else:
            rank = random.sample(range(0,62),2)    # EMNIST数据集，有62个标签
            #indices = [[] for i in range(62)]

        # 按照数据的标签划分成10组，每个组存放相应的索引
        for index, data in enumerate(self.train_dataset):
            if data[1] in rank:
                dataset_indices.append(index)

        for index, data in enumerate(self.test_dataset):
            if data[1] in rank:
                testdata_indices.append(index)
            # if data[1] not in label:      # 标签不存在，分配一个标签，以及一个编号
            #     label.append(data[1])
            #     indices[i].append(index)
            #     i = i + 1
            # else:
            #     l = label.index(data[1])  # 标签存在，在相应的indices中添加新的索引
            #     indices[l].append(index)

        print(rank)
        dataset1 = torch.utils.data.Subset(self.train_dataset, dataset_indices)    # 获取指定元素的数据集
        dataset2 = torch.utils.data.Subset(self.test_dataset, testdata_indices)    # 获取指定元素的数据集
        train_dataset = DataLoader(dataset1, batch_size=self.batch_size, shuffle=True,num_workers=8,pin_memory=True)
        test_dataset = DataLoader(dataset2, batch_size=self.batch_size, shuffle=True,num_workers=8,pin_memory=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data


    # 获得practical_nonIID_data，这个更加符合实际
    # 将所有的数据分为三个组，每个组内的数据不完全一样。
    # 每个组具有80%的重要数据，20%的其他数据
    def get_practical_nonIID_data(self,train_num=0.02,test_num=1):
        mnist_label = [[0,1,2],[3,4,5],[6,7,8,9]]
        cifar_label = [[0,1,2],[3,4,5],[6,7,8,9]]
        emnist_label = [[0,1,2,3,4,5,6,7,8,9],         # 数字集合，大写字母集合，小写字母集合
                        [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
                        [36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]]

        data_indices = [[],[],[]]
        test_indices = [[],[],[]]
        if self.data_name == 'MNIST':              # 对三个数据集分别划分,每个数据集划分成三组

            for indix,data in enumerate(self.train_dataset):
                if data[1] in mnist_label[0]:
                    data_indices[0].append(indix)
                elif data[1] in mnist_label[1]:
                    data_indices[1].append(indix)
                else:
                    data_indices[2].append(indix)
            for indix,data in enumerate(self.test_dataset):
                if data[1] in mnist_label[0]:
                    test_indices[0].append(indix)
                elif data[1] in mnist_label[1]:
                    test_indices[1].append(indix)
                else:
                    test_indices[2].append(indix)

        elif self.data_name == 'CIFAR10':

            for indix,data in enumerate(self.train_dataset):
                if data[1] in cifar_label[0]:
                    data_indices[0].append(indix)
                elif data[1] in cifar_label[1]:
                    data_indices[1].append(indix)
                else:
                    data_indices[2].append(indix)

            for indix,data in enumerate(self.test_dataset):
                if data[1] in cifar_label[0]:
                    test_indices[0].append(indix)
                elif data[1] in cifar_label[1]:
                    test_indices[1].append(indix)
                else:
                    test_indices[2].append(indix)

        else:

            for indix,data in enumerate(self.train_dataset):
                if data[1] in emnist_label[0]:
                    data_indices[0].append(indix)
                elif data[1] in emnist_label[1]:
                    data_indices[1].append(indix)
                else:
                    data_indices[2].append(indix)
            for indix,data in enumerate(self.test_dataset):
                if data[1] in emnist_label[0]:
                    test_indices[0].append(indix)
                elif data[1] in emnist_label[1]:
                    test_indices[1].append(indix)
                else:
                    test_indices[2].append(indix)

        indices = random.randint(0,30002)       # 随机选择一组作为主要成分
        indices = (indices%99)%3
        #sampling_train = random.sample(range(0, len(self.train_dataset)), int(len(self.train_dataset)*train_num))
        #sampling_train = list(set(np.append(sampling_train,data_indices[indices])))    # 合并 并且去重复值
        dataset = torch.utils.data.Subset(self.train_dataset, data_indices[indices])
        train_dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,num_workers=8,pin_memory=True)

        #sampling_test = random.sample(range(0, len(self.test_dataset)), int(test_num * len(self.test_dataset)))
        test_dataset = torch.utils.data.Subset(self.test_dataset, test_indices[indices])
        test_dataset = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8,pin_memory=True)
        
        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data


    # 第一种方式：IID；第二种获得数据的方式：每个用户随机获得2个标签，成为nonIID；
    # 第三种获得数据的方式，将数据划分为三组，每一组的用户数据相似，组之间数据相似小
    def get_data(self,get_data_way):
        if get_data_way == "IID":
            data = self.get_IID_data()
        elif get_data_way == "nonIID":
            data = self.get_nonIID_data()
        else:  # 划分三组，每一组内的数据相似，组间不相似
            data = self.get_practical_nonIID_data()

        return data
