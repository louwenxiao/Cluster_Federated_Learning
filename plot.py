import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

losses = []           # 每个用户的平均损失
accuracyes = [[],[]]  # 第一个元素为每个用户的平均精度，第二个为全局模型的精度
#"IID","NonIID","pNonIID"

def plot_acc(acc,data,dataset):

    dt = pd.DataFrame(acc)

    dt.to_excel("./result/acc_{}.xlsx".format(data), index=0)
    dt.to_csv("./result/acc_{}.csv".format(data), index=0)

    plt.title('{},30 clients'.format(dataset))
    plt.xlabel("epoches")
    plt.ylabel("acc")
    x=np.arange(0,len(acc[0]))
    x[0]=1
    my_x_ticks = np.arange(0, 101, 20)
    plt.xticks(my_x_ticks)
    plt.plot(x,acc[0],label='IID_P')
    plt.plot(x,acc[1],label='IID_GL')
    plt.plot(x,acc[2],label='IID_GA')
    plt.legend()
    plt.savefig('./result/acc_{}_{}.jpg'.format(dataset,data))
    plt.show()
    plt.clf()

