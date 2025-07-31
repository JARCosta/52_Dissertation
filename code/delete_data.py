
import utils
from datasets import datasets



for dataname in datasets.keys():
    for n_neighs in range(5, 16):
        print(f"Deleting {dataname} {n_neighs}")
        utils.pop_measure(dataname, "mvu", n_neighs)
        utils.pop_measure(dataname, "mvu.based", n_neighs)