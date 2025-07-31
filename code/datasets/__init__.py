import numpy as np


# Dataset modules
from . import swiss
from . import s_curve
from . import moons
from . import artificial
from . import natural
from . import sklearn_datasets

from utils import stamp


datasets = {
    "swiss": {
        "func": swiss.default,
        "#components": 2,
    },
    "helix": {
        "func": artificial.helix,
        "#components": 2,
    },
    "twinpeaks": {
        "func": artificial.twinpeaks,
        "#components": 2,
    },
    "broken.swiss": {
        "func": swiss.broken,
        "#components": 2,
    },
    "difficult": {
        "func": artificial.difficult,
        "#components": 5,
    },


    "mnist": {
        "func": natural.mnist,
        "#components": 20,
        "natural": True,
    },
    "coil20": {
        "func": natural.coil20,
        "#components": 5,
        "natural": True,
    },
    "orl": {
        "func": natural.orl,
        "#components": 8,
        "natural": True,
    },
    "hiva": {
        "func": natural.hiva,
        "#components": 15,
        "natural": True,
    },
    "mit-cbcl": {
        "func": natural.mit_cbcl,
        "#components": 6, # TODO: confirm
        "natural": True,
    },


    "parallel.swiss": {
        "func": swiss.parallel,
        "#components": 2,
    },
    "broken.s_curve": {
        "func": s_curve.broken,
        "#components": 2,
    },
    "four.moons": {
        "func": moons.four,
        "#components": 2, #3 # TODO: confirm
    },
    "two.swiss": {
        "func": swiss.two,
        "#components": 2, #3 # TODO: confirm
    },



    "parallel.swiss": {
        "func": swiss.parallel,
        "#components": 2,
    },

    "s_curve": {
        "func": s_curve.default,
        "#components": 2,
    },
    "broken.s_curve": {
        "func": s_curve.broken,
        "#components": 2,
    },

    "moons": {
        "func": moons.default,
        "#components": 2,
    },
    "four.moons": {
        "func": moons.four,
        "#components": 2,
    },
}

def get_dataset(dataname:str, n_points:int, noise:float, random_state:int=None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if random_state is not None:
        np.random.seed(random_state)
    
    if "natural" in datasets[dataname]:
        X, labels, t = datasets[dataname]['func']()
        stamp.print_set(f"*\t {dataname} dataset loaded {X.shape}.")
    else:
        X, labels, t = datasets[dataname]['func'](n_points, noise)
        stamp.print_set(f"*\t {dataname} dataset generated ({X.shape}, noise={noise}, random_state={random_state}).")

    # from plot import plot
    # plot(X, c=labels, block=True, title=f"{dataname} {n_points} points")
    # breakpoint()

    X = X - X.mean(0)
    return X, labels, t