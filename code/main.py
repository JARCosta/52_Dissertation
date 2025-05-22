import datetime
import json
import multiprocessing
import multiprocessing.managers
import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold
from scipy.sparse.csgraph import connected_components

import models
import measure
import utils
from generate_data import get_dataset
from utils import stamp
from plot import plot

def model_func(threads_return:multiprocessing.managers.SyncManager, X:np.ndarray, labels:np.ndarray, model_args:dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    model_args['start'] = datetime.datetime.now().strftime("%d-%H:%M:%S")
    print(f"Running {model_args['model']} on {model_args['dataname']} with {model_args['#neighs']} neighbors")
    Y = models.run(X, model_args)

    if Y is None:
        utils.warning(f"could not compute Y for {model_args['model']} on {model_args['dataname']}")
        return None
    if Y.shape[1] != model_args['#components']:
        utils.warning(f"Y has {Y.shape[1]} dimensions, expected {model_args['#components']}")
    if model_args['plotation']:
        plot(Y, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']} {model_args['#neighs']} neighbors")
    stamp.set()
    One_nn = measure.one_NN(Y, labels)
    stamp.print_set(f"*\t 1-NN \t {One_nn}")
    T, C = measure.TC(X, Y, model_args["#neighs"])
    stamp.print_set(f"*\t T, C \t {T}, {C}")
    
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
        plt.show()

def best_args(threads_return:multiprocessing.managers.DictProxy, labels:np.ndarray) -> tuple[int, dict]:
    if len(threads_return) == 0:
        print("No results")
        return
    results = []
    for model_args, _, One_nn, T, C in threads_return.values():
        results.append([model_args['#neighs'], One_nn, T, C])
    results.sort(key=lambda x: x[0])
    results = np.array(results).T

    results[2:] = np.round(np.log10(1 - results[2:].astype(float) + 1e-10), 3)

    TC = results[2]
    if len(np.where(TC != -np.inf)[0]) > 0:
        best = np.min(TC[np.where(TC != -np.inf)])
    else:
        best = -4
    TC[np.where(TC == -np.inf)] = best - 1


    print(results)
    
    results[2:] = np.round((1 - 10**(results[2:])).astype(float), 3)

    k_best, k_best_measures, k_best_measures_sum = None, None, -np.inf
    for k in results.T:
        if k[1] is not None:
            k_sum = np.sum(1 - k[1] + k[2] + k[3])
            # print(f"k={k[0]}\t -{k[1]} + {k[2]} + {k[3]} = {k_sum}")
        else:
            k_sum = np.sum(k[2] + k[3])
            # print(f"k={k[0]}\t {k[2]} + {k[3]} = {k_sum}")
        if k_sum > k_best_measures_sum:
            # print(f"new best: {k_sum}(k={k[0]})")
            k_best = int(k[0])
            k_best_measures_sum = k_sum
            k_best_measures = {"1-NN": k[1], "T": k[2], "C": k[3]}
    
    plot_args(results, model_args, k_best) # Warning: model_args defined inside the loop (last thread computed)
    
    print(f"Best {model_args['dataname']} {model_args['model']} {k_best_measures}, k={k_best}")
    with open("measures.best.csv", "a") as f:
        print(model_args)
        print(k_best)
        print(k_best_measures)
        f.write(f"{model_args['paper']},{model_args['dataname']},{model_args['model']},{model_args['#points']},{k_best},{k_best_measures['1-NN']},{k_best_measures['T']},{k_best_measures['C']}\n")

    if model_args['plotation']:
        plot(threads_return[k_best][1], c=labels, block=True, title=f"Best {model_args['dataname']} {model_args['model']} {k_best_measures}")


def model_launcher(model_args:dict, models:list, threaded:bool, plotation:bool, verbose:bool, X:np.ndarray, labels:np.ndarray, t:np.ndarray):
    
    for model in models:
        model_args['model'] = model

        threads:list[multiprocessing.Process] = []
        manager = multiprocessing.Manager()
        threads_return = manager.dict()
        for n_neighbors in range(5, 16):
            model_args['#neighs'] = n_neighbors

            # TODO: four.moons, two.swiss, mit-cbcl
            if model_args['dataname'] in ["mnist"]:
                model_args["#components"] = 20
            elif model_args['dataname'] in ["hiva", "nisis"]:
                model_args["#components"] = 15
            elif model_args['dataname'] in ["orl",]:
                model_args["#components"] = 8
            elif model_args['dataname'] in ["difficult", "coil20"]:
                model_args["#components"] = 5
            elif model_args['dataname'] in ["swiss", "twinpeaks", "broken.swiss", "paralell.swiss", "broken.s_curve", ]:
                model_args["#components"] = 2
            elif model_args['dataname'] in ["helix",]:
                model_args["#components"] = 1
            elif model_args['dataname'] in ["four.moons", "two.swiss"]:
                model_args["#components"] = 2 #3 # TODO: confirm
            else:
                raise ValueError(f"Unknown dataset {model_args['dataname']} for model {model}")

            model_args['eps'] = 1e-3 if np.any([i in model for i in ["mvu", "mvu.ineq", "eng", "adaptative", "adaptative2"]]) else None
            model_args['plotation'] = plotation
            model_args['verbose'] = verbose
            model_args = {k: v for k, v in model_args.items() if v is not None}

            if threaded:
                t = multiprocessing.Process(target=model_func, args=[threads_return, X, labels, model_args])
                threads.append(t)
                print(f"launching sub-thread {t}")
                t.start()
            else:
                model_func(threads_return, X, labels, model_args)
        if threaded:
            [t.join() for t in threads]
            print("joined")
        
        best_args(threads_return, labels)


def main(paper:str, model_list:list, datasets:list, n_points:int, threaded:bool, plotation:bool, verbose:bool) -> None:
    model_args = {}
    model_args["paper"] = paper
    model_args["#points"] = n_points

    threads:list[multiprocessing.Process] = []

    for dataname in datasets:
        model_args['dataname'] = dataname
        X, labels, t = get_dataset({'model': "set", 'dataname': dataname, "#points": n_points}, cache=False, random_state=11)
        # plot(X, c=t[:, 0], block=True, title=f"{dataname} {n_points} points")
        
        if threaded:
            t = multiprocessing.Process(target=model_launcher, args=[model_args, model_list, threaded, plotation, verbose, X, labels, t])
            threads.append(t)
            print(f"launching thread     {t}")
            t.start()
        else:
            model_launcher(model_args, model_list, threaded, plotation, verbose, X, labels, t)
    if threaded:
        [t.join() for t in threads]
        print("joined")
    
    if models.mvu.eng is not None:
        print("quitting MATLAB engine")
        models.mvu.eng.quit()
    
    print("finished")



