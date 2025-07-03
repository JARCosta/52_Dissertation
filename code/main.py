from utils import stamp

import datetime
import multiprocessing
import multiprocessing.managers
import os
import matplotlib.pyplot as plt
import numpy as np
stamp.print_set(f"*\t initialization\t {os.path.basename(__file__)}.libs")

import models
import utils
from datasets import get_dataset, datasets
import plot
stamp.print_set(f"*\t initialization\t {os.path.basename(__file__)}.files")

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

def model_func(threads_return:multiprocessing.managers.SyncManager, X:np.ndarray, labels:np.ndarray, model_args:dict, measure_bool:bool) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    utils.important(f"Running {model_args['model']} on {model_args['dataname']} with {model_args['#neighs']} neighbors")
    Y = models.run(X, model_args)

    if Y is None:
        utils.warning(f"could not compute Y for {model_args['model']} on {model_args['dataname']}(k={model_args['#neighs']})")
        return None
    if Y.shape[1] != model_args['#components'] and model_args["#components"] is not None:
        raise ValueError(f"Y has {Y.shape[1]} dimensions, intrinsic is {model_args['#components']}.")

    if model_args['plotation']:
        plot.plot_scales(X, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Original data")
        plot.plot_scales(Y, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Embedding", legend=model_args)
    
    One_nn, T, C = utils.compute_measures(X, Y, labels, model_args["#neighs"])
    if measure_bool:
        utils.store_measure(model_args, One_nn, T, C)
    threads_return[model_args['#neighs']] = (model_args, Y, One_nn, T, C)

def plot_args(results:np.ndarray, model_args:dict, k_best:int) -> None:
    fig, ax1 = plt.subplots()
    plt.title(f"{model_args['model']} applied on {model_args['dataname']}, selected k={k_best}")

    ax1.set_xlabel('size of k-neighbourhood')
    ax1.set_ylabel('log10 of T and C')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel('1-NN')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y')

    
    ax1.plot(results[0], results[2], color='tab:red')
    ax1.plot(results[0], results[3], color='tab:orange')
    ax2.plot(results[0], results[1], color='tab:blue')
    fig.legend(["T", "C", "1-NN"])
    
    ax1.scatter(results[0], results[2], color='tab:red')
    ax1.scatter(results[0], results[3], color='tab:orange')
    ax2.scatter(results[0], results[1], color='tab:blue')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    os.makedirs("cache") if not os.path.exists("cache") else None
    os.makedirs(f"cache/{model_args['model']}") if not os.path.exists(f"cache/{model_args['model']}") else None
    plt.savefig(f"cache/{model_args['model']}/{model_args['dataname']}.png")
    if model_args['plotation']:
        plt.show(block=False)

def best_args(results:np.ndarray, model_args:dict, labels:np.ndarray) -> tuple[int, dict]:
    results.sort(key=lambda x: x[0])
    results = np.array(results).T

    if len(results) == 0:
        print("No results")
        return

    results[2:] = np.round(np.log10(1 - results[2:].astype(float) + 1e-10), 3)
    TC = results[2]
    if len(np.where(TC != -np.inf)[0]) > 0:
        best = np.min(TC[np.where(TC != -np.inf)])
    else:
        best = -4
    TC[np.where(TC == -np.inf)] = best - 1
    results[2:] = np.round((1 - 10**(results[2:])).astype(float), 3)



    best_measure_sum = -np.inf
    best_k, best_1_NN, best_T, best_C = None, None, None, None
    for k in results.T:
        # k_sum = np.sum(1 - k[1] + k[2] + k[3]) if k[1] is not None else np.sum(k[2] + k[3])
        k_sum = 1 - k[1]
        if k_sum > best_measure_sum:
            best_measure_sum = k_sum
            best_k, best_1_NN, best_T, best_C = int(k[0]), float(k[1]), float(k[2]), float(k[3])
    if model_args['plotation']:
        plot_args(results, model_args, best_k) # Warning: model_args defined inside the loop (last thread computed)
    
    print(f"Best {model_args['dataname']} {model_args['model']} {best_1_NN}, {best_T}, {best_C}, k={best_k}")
    model_args['#neighs'] = best_k
    utils.store_measure(model_args, best_1_NN, best_T, best_C, best=True)

    if model_args['plotation']: # The best model might not have been computed in this execution (can't get Y)
        # plot.plot(threads_return[best_k][1], c=labels, block=True, title=f"Best {model_args['dataname']} {model_args['model']} {best_1_NN}, {best_T}, {best_C}")
        pass

def model_launcher(model_args:dict, models:list, X:np.ndarray, labels:np.ndarray, t:np.ndarray, threaded:bool, plotation:bool, verbose:bool, measure:bool, pause:bool):
    
    for model in models:
        model_args['model'] = model

        threads:list[multiprocessing.Process] = []
        manager = multiprocessing.Manager()
        threads_return = manager.dict()
        for n_neighbors in range(5, 16):
            model_args['#neighs'] = n_neighbors
            model_args['#components'] = datasets[model_args['dataname']]['#components']
            
            if np.any([i in model for i in ["mvu", "mvu.ineq", "eng", "adaptative", "adaptative2"]]):
                model_args['eps'] = datasets[model_args['dataname']]['eps']

            model_args['plotation'] = plotation
            model_args['verbose'] = verbose
            model_args = {k: v for k, v in model_args.items() if v is not None}

            if threaded:
                t = multiprocessing.Process(target=model_func, args=[threads_return, X, labels, model_args, measure])
                threads.append(t)
                print(f"launching sub-thread {t}")
                t.start()
            else:
                model_func(threads_return, X, labels, model_args, measure)
                if pause or model_args['plotation']:
                    input("Press Enter to continue...")
        if threaded:
            [t.join() for t in threads]
            print("joined")
        
        measures = utils.get_measures(model_args['dataname'], model_args['model'])
        results = []
        for k_neigh in measures.keys():
            results.append([int(k_neigh), measures[k_neigh]['1-NN'], measures[k_neigh]['T'], measures[k_neigh]['C']])
        # results[:] = [k_neigh, One_nn, T, C]

        if "plotation" not in model_args:
            model_args['plotation'] = plotation
        if "verbose" not in model_args:
            model_args['verbose'] = verbose
        best_args(results, model_args, labels)

        if threaded and pause:
            input("Press Enter to continue...")


def main(paper:str, model_list:list, dataset_list:list, n_points:int, threaded:bool, plotation:bool, verbose:bool, measure:bool, pause:bool, seed:int, noise:float) -> None:
    model_args = {}
    model_args["threaded"] = threaded
    model_args["paper"] = paper

    threads:list[multiprocessing.Process] = []

    for dataname in dataset_list:
        model_args['dataname'] = dataname
        X, labels, t = get_dataset(dataname, n_points, noise, random_state=seed)
        # X, labels, t = generate_data.get_dataset({'model': "set", 'dataname': dataname, "#points": n_points}, random_state=11)
        model_args["#points"] = X.shape[0]
        # plot(X, c=t[:, 0], block=True, title=f"{dataname} {n_points} points")
        
        if threaded:
            t = multiprocessing.Process(target=model_launcher, args=[model_args, model_list, X, labels, t, threaded, plotation, verbose, measure, pause])
            threads.append(t)
            print(f"launching thread     {t}")
            t.start()
        else:
            model_launcher(model_args, model_list, X, labels, t, threaded, plotation, verbose, measure, pause)
    if threaded:
        [t.join() for t in threads]
        print("joined")
    
    if models.mvu.eng is not None:
        print("quitting MATLAB engine")
        models.mvu.eng.quit()
    
    print("finished")



