import numpy as np
import config as conf
from concurrent.futures import ProcessPoolExecutor, wait


def grouping(n, group_size):
    clients = list(range(n))
    # np.random.shuffle(clients)
    group_num = int(n / group_size)

    groups = []
    for g in range(group_num):
        if g == group_num - 1:
            start = g * group_size
            end = n
            groups.append(clients[start:end])
        else:
            start = g * group_size
            end = (g + 1) * group_size
            groups.append(clients[start:end])

    user_group = [None for x in range(conf.n)]  # users' group id
    for g in range(group_num):
        for user in groups[g]:
            user_group[user] = g

    return groups, user_group


def PSU(cenServer, thiParServer, clients):

    groups, user_group = grouping(conf.n, conf.group_size)

    """clients' process"""
    with ProcessPoolExecutor(max_workers=conf.processes) as executor:
        futures = []
        for user in clients:
            future = executor.submit(user.masking_itemVec)
            futures.append(future)
        wait(futures)
        for future, user in zip(futures, clients):
            user.get_maskItemVec_result(future.result())

    """Third-Party Server's process"""
    itemVec_mask = thiParServer.itemVecMask_aggregation(clients, groups)

    """Central Server's process"""
    cenServer.itemVec_aggregation(clients, itemVec_mask, groups)

    """clients' process"""
    for user in range(conf.n):
        group_id = user_group[user]
        clients[user].download_Igroup(cenServer.Igroups[group_id])

    """
    The objects of clients will be copied when multiprocessing, 
    thus clear some useless cache will accelerate multiprocessing.
    """
    for user in clients:
        user.clear_cache_psu()
