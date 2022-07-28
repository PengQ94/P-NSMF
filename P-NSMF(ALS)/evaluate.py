import config as conf
import multiprocessing as mp
from output_func import write_result


def evaluate(clients, V, t):

    # evaluate in each client
    for user in clients:
        res = user.evaluate(V)
        user.get_evaluate_result(res)

    # multiprocessing also works, but sometimes costs more time.
    """
    pool = mp.Pool(conf.processes)
    results = []
    for user in clients:
        res = pool.apply_async(user.evaluate, args=(V, t))
        results.append(res)
    pool.close()
    pool.join()
    for res, user in zip(results, clients):
        user.get_evaluate_result(res.get(), t)
    """

    num_test_user = 0
    precision_sum = 0
    recall_sum = 0
    F1_sum = 0
    NDCG_sum = 0
    oneCall_sum = 0

    # get test results from clients
    for user in clients:
        if user.is_test == 1:
            num_test_user += user.is_test
            precision_sum += user.precision
            recall_sum += user.recall
            F1_sum += user.F1
            NDCG_sum += user.NDCG
            oneCall_sum += user.oneCall

    precision = "{:.8f}".format(precision_sum / num_test_user)
    recall = "{:.8f}".format(recall_sum / num_test_user)
    F1 = "{:.8f}".format(F1_sum / num_test_user)
    NDCG = "{:.8f}".format(NDCG_sum / num_test_user)
    oneCall = "{:.8f}".format(oneCall_sum / num_test_user)

    write_result(conf.result_file, [t, precision, recall, F1, NDCG, oneCall])
