import time
import config as conf
from concurrent.futures import ProcessPoolExecutor, wait
from evaluate import evaluate
import output_func


def train(cenServer, thiParServer, clients):
    # create a file to store test results
    output_func.create_result_file(conf.result_file, str(vars(conf.args)))

    executor = ProcessPoolExecutor(max_workers=conf.processes)

    print("Start training!")

    for t in range(1, conf.T+1):

        start = time.time()

        """clients' process"""
        for user in clients:
            user.downlaod_para(cenServer.SV, cenServer.V[user.Igroup])

        futures = []
        for user in clients:
            future = executor.submit(user.update)
            futures.append(future)
        wait(futures)
        for future, user in zip(futures, clients):
            user.get_update_result(future.result())

        """Third-Party Server's process"""
        bU_mask, AU_tri_mask, SU_mask = thiParServer.paraMask_aggregation(clients)

        """Central Server's process"""
        cenServer.para_aggregation(clients, bU_mask, AU_tri_mask, SU_mask)
        cenServer.update()

        """
        The objects of clients will be copied when multiprocessing, 
        thus clear some useless cache will accelerate multiprocessing.
        """
        for user in clients:
            user.clear_cache()

        """Evaluation"""
        if t % 10 == 0:
            evaluate(clients, cenServer.V, t)

        print(f"Iteration {t} used time (s):", time.time() - start)
    
    print("Finished!")
