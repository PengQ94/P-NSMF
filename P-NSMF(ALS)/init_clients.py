from client import Client
import config as conf
import numpy as np


def init_clients():
    # 初始化客户端
    clients = []
    for user in range(conf.n):
        clients.append(Client())

    # 初始化客户端训练数据
    train_data = open(conf.train_path)
    for row in train_data:
        r = row.split(' ')
        user = int(r[0]) - 1
        item = int(r[1]) - 1
        clients[user].Iu.append(item)

    # 没交互的物品
    I = np.arange(conf.m)
    for user in clients:
        user.Iu.sort()
        user.I_sub_Iu = np.setdiff1d(I, user.Iu, assume_unique=True)

    # 初始化客户端测试数据
    test_data = open(conf.test_path)
    for row in test_data:
        r = row.split(' ')
        user = int(r[0]) - 1
        item = int(r[1]) - 1
        clients[user].Iu_test.append(item)

    return clients
