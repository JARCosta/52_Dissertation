import argparse
from main import main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the model launcher.")
    parser.add_argument("--paper", type=str, default=["comparative", "eng"], help="Paper to use for the model launcher.")
    parser.add_argument("--n_points", type=int, default=3000, help="Number of points to use for the model launcher.")
    parser.add_argument("--threaded", action='store_true', default=False, help="Use threading for the model launcher.")
    args = parser.parse_args()

    if args.paper == "comparative":
        models = [
            "pca", 
            "isomap",
            # "mvu",
            "lle",
            "le",
            "hlle",

            # Sub
            "isomap.skl",
            "lle.skl",
            "le.skl",
            "hlle.skl",
            "ltsa.skl",
        ]
        datasets = [
            'swiss',
            'helix',
            'twinpeaks',
            'broken.swiss',
            'difficult',

            'mnist', # TODO: memory issues
            'coil20',
            'orl',
            # 'nisis',
            # 'hiva',
        ]

    elif args.paper == "eng":
        models = [
            "isomap",
            "isomap.eng",
            "lle",
            "lle.eng",
            "le",
            "le.eng",
            "hlle",
            "hlle.eng",
            # "jme",
            
            # Sub
            "lle.skl",
            "le.skl",
            "hlle.skl",    
        ]
        datasets = [
            'broken.swiss',
            'paralell.swiss',
            'broken.s_curve',
            'four.moons',
            'two.swiss',
            'coil20', 
            # 'mit-cbcl', # TODO: import
        ]

    elif args.paper == "dev":
        models = [
            "mvu",
            # "mvu.eng",
        ]
        datasets = [
            'swiss',
            'helix',
            'twinpeaks',
            'broken.swiss',
            'difficult',

            # 'mnist', # TODO: memory issues, ada1 can't load dataset
            'coil20',
            'orl',
            # 'nisis',
            # 'hiva',
            ]

    elif args.paper == "none":
        datasets = [
        ]
        for dataname in datasets:
            X, labels, t = get_dataset({'model': "set", 'dataname': dataname, "#points": n_points}, cache=False, random_state=11)
            None_1_NN = measure.one_NN(X, labels)
            with open("cache/measures.csv", "a") as f:
                f.write(f"{dataname},none,{n_points},None,{None_1_NN},None,None\n")
        input("finished?")

    else:
        raise ValueError("Paper not found. Please use 'comparative' or 'eng'.")
        datasets = [
            'teapots', # TODO: needs smaller n_neighbors
            'swiss_toro',

            'changing.swiss',
            '3d_clusters',
        ]
    
    main(
        paper=args.paper,
        datasets=datasets,
        models=models,
        n_points=args.n_points,
        threaded=args.threaded,
    )

