import argparse

from main import main
from datasets import get_dataset
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the model launcher.")
    parser.add_argument("--paper", type=str, default=["comparative", "eng"], help="Paper to use for the model launcher.")
    parser.add_argument("--n_points", type=int, default=2000, help="Number of points to use for the model launcher.")
    parser.add_argument("--threaded", action='store_true', default=False, help="Use threading for the model launcher.")
    parser.add_argument("--plotation", action='store_true', default=False, help="Plot the results.")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose output.")
    parser.add_argument("--measure", action='store_true', default=False, help="Store the measure's results.")
    parser.add_argument("--pause", action='store_true', default=False, help="Pause for every model execution.")
    parser.add_argument("--seed", type=int, default=27, help="Random seed.")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level for the dataset.")


    args = parser.parse_args()

    if args.paper == "comparative":
        models = [
            # "pca", 
            # "isomap",
            # "lle",
            # "le",
            # "hlle",

            # Sub
            "isomap.skl",
            "lle.skl",
            "le.skl",
            "hlle.skl",
            "ltsa.skl",
            
            # "mvu",
        ]
        dataset_list = [
            'swiss',
            'helix',
            'twinpeaks',
            'broken.swiss',
            'difficult',

            # 'mnist', # TODO: memory overload
            'coil20',
            'orl',
            # 'nisis', # Does not exist
            # 'hiva', # TODO: problems on the import
        ]

    elif args.paper == "eng":
        models = [
            # "isomap",
            # "isomap.eng",
            # "lle",
            # "lle.eng",
            # "le",
            "le.eng",
            "hlle",
            "hlle.eng",
            # "jme",
            
            # Sub
            "lle.skl",
            "le.skl",
            "hlle.skl",    
        ]
        dataset_list = [
            # 'broken.swiss',
            # 'parallel.swiss',
            # 'broken.s_curve',
            'four.moons',
            # 'two.swiss',
            # 'coil20', 
            # 'mit-cbcl', # TODO: import
        ]

    elif args.paper == "mvu":
        models = [
            # "mvu",
            # "mvu.eng",
            "mvu.our",
        ]
        dataset_list = [
            # 'swiss',
            # 'helix',
            # 'twinpeaks', # Mostly NaNs
            # 'broken.swiss',
            # 'difficult',

            # 'mnist', # TODO: memory overload
            'coil20', # TODO: 1440 points 1396 components found
            'orl',
            # 'nisis', # Does not exist
            # 'hiva', # TODO: problems on the import


            'parallel.swiss',
            'broken.s_curve', # full NaNs
            'four.moons',
            'two.swiss',
            # 'mit-cbcl', # TODO: import
            ]
    elif args.paper == "dev":
        models = [
            # # Comparitive
            # "pca", 
            "isomap",
            # "lle",
            # "le",
            # "hlle",

            # # Comparitive-Sub
            # "isomap.skl",
            # "lle.skl",
            # "le.skl",
            # "hlle.skl",
            # "ltsa.skl",
            

            # # Eng
            # "isomap.eng",
            # "lle.eng",
            # "le.eng",
            # "hlle.eng",
            # # "jme",
            
            # 'mvu',
        ]
        dataset_list = [
            # 'swiss',
            # 'helix',
            # 'twinpeaks',
            # 'broken.swiss',
            # 'difficult',

            # # 'mnist', # TODO: memory issues, ada1 can't load dataset
            # 'coil20',
            'orl',
            # 'nisis',
            # 'hiva',

            'parallel.swiss',
            # 'broken.s_curve.4', # easiest
            'broken.s_curve', # default
            # 'broken.s_curve.1', # most class changes
            'four.moons',
            'two.swiss',
            # 'mit-cbcl', # TODO: import
            
            # 'teapots',
            ]

    elif args.paper == "none":
        dataset_list = [
            'broken.swiss',
            'parallel.swiss',
            'broken.s_curve',
            'four.moons',
            'two.swiss',
            'coil20', 
            # 'mit-cbcl', # TODO: import
        ]
        for dataname in dataset_list:
            X, labels, t = get_dataset(dataname, args.n_points, args.noise, random_state=args.seed)
            None_1_NN = utils.one_NN(X, labels)
            print(f"loaded {dataname} {None_1_NN}")
            utils.store_measure({'dataname': dataname, 'model': 'none', '#neighs': None, '#points': X.shape[0]}, None_1_NN, best=True)
            utils.store_measure({'dataname': dataname, 'model': 'none', '#neighs': None, '#points': X.shape[0]}, None_1_NN)
            
            # with open("measures.best.csv", "a") as f:
            #     f.write(f"none,{dataname},none,{args.n_points},None,{None_1_NN},None,None\n")
            # with open("measures.all.csv", "a") as f:
            #     f.write(f"none,{dataname},none,{args.n_points},None,{None_1_NN},None,None\n")
        
        exit()

    else:
        raise ValueError("Paper not found. Please use 'comparative' or 'eng'.")
        datasets = [
            'teapots', # TODO: needs smaller n_neighbors
            'swiss_toro',

            'changing.swiss',
            '3d_clusters',
        ]
    
    try:
        main(
            paper=args.paper,
            model_list=models,
            dataset_list=dataset_list,
            n_points=args.n_points,
            threaded=args.threaded,
            plotation=args.plotation if not args.threaded else False,
            verbose=args.verbose if not args.threaded else False,
            measure=args.measure,
            pause=args.pause if not args.threaded else False,
            seed=args.seed,
            noise=args.noise,
        )
    finally:
        utils.stamp.print("* Killing process")
        from models.mvu import eng
        if eng is not None:
            eng.quit()

