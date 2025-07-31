import traceback
import datetime
import multiprocessing
import multiprocessing.managers
import os
import matplotlib.pyplot as plt
import numpy as np

import models
import utils
from datasets import get_dataset, datasets
import plot

utils.stamp.print_set(f"*\t initialization\t {os.path.basename(__file__)}.files")

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

def model_func(threads_return:multiprocessing.managers.DictProxy, X:np.ndarray, labels:np.ndarray, model_args:dict, measure_bool:bool) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    utils.important(f"Running {model_args['model']} on {model_args['dataname']} with {model_args['#neighs']} neighbors")
    Y = models.run(X, model_args, labels)

    if Y is None:
        utils.warning(f"could not compute Y for {model_args['model']} on {model_args['dataname']}(k={model_args['#neighs']})")
        utils.pop_measure(model_args['dataname'], model_args['model'], model_args['#neighs'])
        return None
    
    # if model_args["#components"] is not None:
    #     if Y.shape[1] > model_args['#components']:
    #         utils.warning(f"Y has {Y.shape[1]} dimensions, intrinsic is {model_args['#components']}. Restricting to {model_args['#components']} components")
    #         Y = Y[:, :model_args['#components']]
    #     elif Y.shape[1] < model_args['#components']:
    #         utils.warning(f"Y has {Y.shape[1]} dimensions, intrinsic is {model_args['#components']}. Increasing to {model_args['#components']} components")
    #         Y = np.concatenate([Y, np.zeros((Y.shape[0], model_args['#components'] - Y.shape[1]))], axis=1)

    if model_args['plotation']:
        plot.plot_scales(X, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Original data")
        plot.plot_scales(Y, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Embedding", legend=model_args)
        if Y.shape[1] > 3:
            plot.plot_scales(Y[:, 3:], c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Embedding, 3-D", legend=model_args)

    if 'labels' in model_args:
        labels = model_args['labels']
    One_nn, T, C = utils.compute_measures(X, Y, labels, model_args["#neighs"])
    if measure_bool:
        utils.add_measure(model_args, Y.shape[1], One_nn, T, C)
    threads_return[model_args['#neighs']] = (model_args, Y, One_nn, T, C)

def model_launcher(model_args:dict, models:list, X:np.ndarray, labels:np.ndarray, t:np.ndarray, threaded:bool, plotation:bool, verbose:bool, measure:bool, pause:bool):
    
    for model in models:
        model_args['model'] = model

        threads:list[multiprocessing.Process] = []
        manager = multiprocessing.Manager()
        threads_return = manager.dict()
        for n_neighbors in range(model_args['k_small'], model_args['k_large'] + 1):
            model_args['#neighs'] = n_neighbors
            model_args['#components'] = datasets[model_args['dataname']]['#components']
            
            model_args['plotation'] = plotation
            model_args['verbose'] = verbose
            model_args = {k: v for k, v in model_args.items() if v is not None}
            model_args['status'] = None
            model_args['restricted'] = None
            model_args['artificial_connected'] = None


            if threaded:
                process = multiprocessing.Process(target=model_func, args=[threads_return, X, labels, model_args, measure])
                threads.append(process)
                print(f"launching sub-thread {process}")
                process.start()
            else:
                model_func(threads_return, X, labels, model_args, measure)
                if pause or plotation:
                    input("Press Enter to continue...")
                    plt.close('all')
        if threaded:
            [t.join() for t in threads]
            print("joined")

        if measure:
            utils.plot_measures(model_args['dataname'], model, model_args['plotation'])
        
        if threaded and pause:
            input("Press Enter to continue...")


def main(paper:str, model_list:list, dataset_list:list, n_points:int, k_small:int, k_large:int, threaded:bool, plotation:bool, verbose:bool, measure:bool, pause:bool, seed:int, noise:float) -> None:
    model_args = {}
    model_args["k_small"] = k_small
    model_args["k_large"] = k_large
    model_args["threaded"] = threaded
    model_args["paper"] = paper

    for dataname in datasets.keys():
        for model in model_list:
            utils.plot_measures(dataname, model)
    
    threads:list[multiprocessing.Process] = []

    for dataname in dataset_list:
        model_args['dataname'] = dataname
        X, labels, t = get_dataset(dataname, n_points, noise, random_state=seed)
        # X, labels, t = generate_data.get_dataset({'model': "set", 'dataname': dataname, "#points": n_points}, random_state=11)
        model_args["#points"] = X.shape[0]
        # plot(X, c=t[:, 0], block=True, title=f"{dataname} {n_points} points")
        
        if threaded:
            process = multiprocessing.Process(target=model_launcher, args=[model_args, model_list, X, labels, t, threaded, plotation, verbose, measure, pause])
            threads.append(process)
            print(f"launching thread     {process}")
            process.start()
        else:
            model_launcher(model_args, model_list, X, labels, t, threaded, plotation, verbose, measure, pause)
    if threaded:
        [t.join() for t in threads]
        print("joined")
    
    if models.mvu.eng is not None:
        print("quitting MATLAB engine")
        models.mvu.eng.quit()
    
    print("finished")



