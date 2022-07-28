import numpy as np
import config as conf
import csv
import time

Iu = [[] for x in range(conf.n)]  # items that interact with user u in train data
Iu_test = [[] for x in range(conf.n)]  # items that interact with user u in test data
Ui = [[] for x in range(conf.m)]  # users who interact with item i in train data

# import train data
train_data = open(conf.train_path)
for row in train_data:
    r = row.split(" ")
    user = int(r[0]) - 1
    item = int(r[1]) - 1
    Iu[user].append(item)
    Ui[item].append(user)
for items in Iu:
    items.sort()
    items = np.array(items)
for users in Ui:
    users.sort()
    users = np.array(users)

# import test data
test_data = open(conf.test_path)
for row in test_data:
    r = row.split(" ")
    user = int(r[0]) - 1
    item = int(r[1]) - 1
    Iu_test[user].append(item)
for items in Iu_test:
    items.sort()

# parameters
U = (np.random.rand(conf.n, conf.d) - 0.5) * 0.01  # user-specific latent feature matrix
V = (np.random.rand(conf.m, conf.d) - 0.5) * 0.01  # item-specific latent feature matrix
omega = conf.omega
beta = conf.beta
alpha = 1
gamma = conf.gamma

# evaluate by precision, recall, F1, NDCG, oneCall (@topK)
def evaluate(t):
    num_test_user = 0.
    precision_sum = 0.
    recall_sum = 0.
    F1_sum = 0.
    NDCG_sum = 0.
    oneCall_sum = 0.

    for user in range(conf.n):
        if len(Iu_test[user]) == 0:
            continue

        num_test_user += 1

        pred = np.matmul(U[user], V.T)
        pred[Iu[user]] = -(1<<20)
        item_topK = np.argpartition(pred, kth=-conf.topK)[-conf.topK:]  # unsorted topK
        pred_topK = pred[item_topK]
        idx_topK = np.argsort(-pred_topK)
        item_topK = list(item_topK[idx_topK])  # sorted topK

        hit = 0.
        DCG = 0.
        DCGbest = 0.

        DCGbest_topK = min(len(Iu_test[user]), conf.topK)
        for k in range(DCGbest_topK):
            DCGbest += 1. / np.log2(k + 2)

        for k in range(conf.topK):
            if item_topK[k] in Iu_test[user]:
                hit += 1.
                DCG += 1. / np.log2(k + 2)

        precision = hit / conf.topK
        recall = hit / len(Iu_test[user])
        F1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        precision_sum += precision
        recall_sum += recall
        F1_sum += F1
        NDCG_sum += DCG / DCGbest
        oneCall_sum += 1 if hit > 0 else 0

    precision = "{:.8f}".format(precision_sum / num_test_user)
    recall = "{:.8f}".format(recall_sum / num_test_user)
    F1 = "{:.8f}".format(F1_sum / num_test_user)
    NDCG = "{:.8f}".format(NDCG_sum / num_test_user)
    oneCall = "{:.8f}".format(oneCall_sum / num_test_user)

    return [t, precision, recall, F1, NDCG, oneCall]


# create files to store test result
with open(conf.result_file, "a", newline="") as f:
    f.write(str(vars(conf.args)) + "\n\n")
    f.write("iter,pre,rec,F1,NDCG,oneCall \n")

# train
print("Start training!")

for t in range(1, conf.T+1):

    start = time.time()

    # update U
    SV = np.matmul(V.T, beta * V)
    for user in range(conf.n):
        V_u = V[Iu[user]]
        pred = np.matmul(U[user], V_u.T)
        coe = (omega - alpha * beta) * pred - omega
        aV = np.matmul(coe, V_u)
        gradU = (aV + np.matmul(alpha * U[user], SV)) * 2 / conf.m
        reg  = 2 * conf.lambda_ * U[user]
        U[user] -= gamma * (gradU + reg)

    # update V
    SU = np.matmul(U.T, alpha * U)
    for item in range(conf.m):
        U_i = U[Ui[item]]
        pred = np.matmul(V[item], U_i.T)
        coe = (omega - alpha * beta) * pred - omega
        aU = np.matmul(coe, U_i)
        gradV = (aU + np.matmul(beta * V[item], SU)) * 2 / conf.n
        reg  = 2 * conf.lambda_ * V[item]
        V[item] -= gamma * (gradV + reg)

    gamma *= 0.999

    print(f"Iteration {t} used time (s):", time.time() - start)

    # evaluate and store test results
    if t % 10 == 0:
        eval_res = evaluate(t)
        with open(conf.result_file, "a", newline="") as f:
            csv_writer = csv.writer(f, delimiter="\t")
            csv_writer.writerow(eval_res)

print("Finished!")