"""
Compute the communication cost
"""

import config as conf


def cal_items_per_user(clients):
    sum_items = 0
    for user in clients:
        sum_items += len(user.Igroup)
    items_per_user = sum_items / conf.n  # average |Iu + I'u|

    return items_per_user


def para_cost_per_user_iter(items_per_user):
    # downlaod para
    para_cost = conf.d ** 2 + items_per_user * conf.d
    # upload para
    para_cost += (
        conf.d ** 2 + items_per_user * (conf.dimension_tri + conf.d)
    ) * 2
    # byte
    para_cost *= 8

    return para_cost


def psu_cost_per_user(items_per_user):
    # upload item vector
    psu_cost = 2 * conf.m
    # downlaod Igroup
    psu_cost += items_per_user
    # byte
    psu_cost = psu_cost * 8

    return psu_cost


def communication_cost(clients):
    items_per_user = cal_items_per_user(clients)
    para_cost = para_cost_per_user_iter(items_per_user)
    psu_cost = psu_cost_per_user(items_per_user)

    with open(conf.cost_file, "a", newline="") as f:
        f.write("{group_size = " + str(conf.group_size) + "}\n")
        # f.write("Average |Iu + I'u| per user: " + str(int(items_per_user)) + "\n")
        f.write(
            "Average communication cost of PSU per user: " + str(int(psu_cost)) + " B\n"
        )
        f.write(
            "Average communication cost of parameters per user per iteration: "
            + str(int(para_cost))
            + " B\n\n"
        )
