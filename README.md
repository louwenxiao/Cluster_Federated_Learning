# Federated Learning with Cluster

## 1.创作目的

	本人硕士一年级在读，专注于边缘计算方向，目前主要关注联邦学习的内容。在解决数据的non-IID过程中，有一个想法，并且用代码做了一个小实验。
	想法：现实世界中，non-IID非常普遍，但是也不是完全非独立同分布的，因为物以类聚人以群分。那么我们能不能用聚类的方法进行学习？将每一个设备上训练的参数上传后，首先进行聚类，将相似的设备分为一类。划分一个个簇以后，每一个簇内进行联邦学习。
	当然，利用聚类的方法解决non-IID问题的论文也有，但是基本都是每次聚合的时候分类，这样每次聚合都用聚类算法代价太高。所以我们希望可以先划分一个个簇，每一个簇内部单独训练。An Efficient Framework for Clustered Federated Learning.（NIPS2020）；ON THE BYZANTINE ROBUSTNESS OF CLUSTERED FEDERATED LEARNING

## 2.文件结构
+ cache  :   存放产生的模型文件

+ clients_and_server

    + __init__.py
    + clients.py   ：产生用户需要用到
    + cluster.py   :  聚类算法
    + server.py    :  产生云服务

+ data  :   下载相应的数据集

+ data_and_model   

    + __init__.py
    + datasets.py    :  产生相应的数据集
    + models.py      :  产生模型

+ result :    存放实验结果

+ main：主程序

+ plot： 画图


## 3.详细描述文件

### 3.1 cache文件夹
      这个文件夹内存放模型训练的结果，包括用户的模型和簇模型

### 3.2 clients_and_server文件夹
      这个文件夹包含三个文件，clients、cluster和server文件。

#### 3.2.1 clients文件
      这个文件用来定义一个用户，一个用户的信息：自身标号、簇编号、模型、训练测试数据、学习率、优化器、训练次数等。包含6个函数：get_cluster_modal、get_model、local_train、pre_train、test_model和updata_clu
      get_cluster_modal：用来获得簇模型。在整体每一轮循环中，簇内部训练L轮会簇内部聚合，簇内部聚合的模型送到簇内的每一个用户。
      get_model：获得全局模型。全局模型训练完成后，将所有的簇模型再平均一下，得到全局模型（这样，不会因为某一类的用户数量过少，导致聚合的时候“话语权”较小），将全局模型送到每一个用户手里，然后开始下一轮训练。
      local_train：模型在本地训练
      pre_train：模型聚类之前，首先需要预训练，获得一个描述本身数据的模型
      test_model：测试模型，返回accuracy
      updata_clu：模型聚类之后，更新所在簇的编号。

#### 3.2.2 cluster文件
      本文件是对用户进行聚类，一共使用三种聚类方式：k均值、层次聚类和密度聚类。另外2个函数load_clients和distance函数，第一个函数是加载所有用户模型，将字典类型变成list类型，第二个函数是计算两个模型之间的距离。
      K_means_cluster，Hierarchical_clustering分别是k均值和层次聚类。Density_clustering表示密度聚类，并没有给出代码。在实验的过程中，我们发现k均值是完全可以满足我们的要求的，而层次聚类不能满足。

#### 3.2.3 server文件
      本文件包含一个云服务类，用于定义云服务端，以及簇模型。
      aggregate_model：用于将所有簇模型聚合
      gain_acc：测试每个簇的精度
      aggregate_cluster：簇内部模型聚合
      
### 3.3 data文件夹
      本文件夹用于存放数据集，本实验可以采用MNIST、CIFAR10、EMNIST和FMNIST数据集

### 3.4 data_and_model文件夹
      这个文件夹内存放2个文件，datasets和model

#### 3.4.1 datasets文件
      本文件产生一个类，用于产生数据。
      __load_dataset函数，用于下载相应的数据，放在data文件夹内。get_IID_data、get_nonIID_data和get_practical_nonIID_data是产生数据的3中方式，第一个是产生IID数据，第二个是产生non-IID数据（每个用户的数据包含两个标签），第三种是比较实际的方式。
      get_data函数，通过这个函数返回获得的数据。

#### 3.4.2 models文件
      本文件产生初始模型。使用CNN模型，CIFAR10和MNIST数据集的模型不一样，直接通过get_model函数返回一个模型。

### 3.5 main文件
      main函数是整个程序的逻辑。首先第一步是数据集的名字使用download_data函数下载数据集，产生一个data_loader的变量用于产生数据，然后根据数据集的名字和模型的种类产生初始模型。产生global_nums个模型，并进行预训练，然后根据已有的模型进行聚类，产生k个簇模型。进行num_glob_iters轮训练，在每一轮训练中，每个用户单独训练，然后簇内部聚合，将所有的簇模型聚合形成全局模型。最后画图。

### 3.6 plot文件
      画图。

### 3.7 result文件夹
      存放结果。








