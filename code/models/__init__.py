# models/__init__.py

from .spectral import *
from .neighbourhood import *

from .mvu import *
from .isomap import *
from .pca import *
from .le import *
from .lle import *
from .ltsa import *

from .extensions import *

def run(X, model_args):
    import sklearn.manifold
    from utils import stamp_set, stamp_print

    stamp_set()

    ########################################################
    # PCA ##################################################
    ########################################################
    
    if model_args['model'].lower() == "pca":

        model = models.pca.PCA(model_args, model_args['#components'])
        Y = model.fit_transform(X)


    ########################################################
    # Isomap ###############################################
    ########################################################

    elif model_args['model'].lower() == "isomap":

        model = models.isomap.Isomap(model_args, model_args["#neighs"], model_args["#components"])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "isomap.skl":

        model = sklearn.manifold.Isomap(n_neighbors=model_args['#neighs'], n_components=model_args['#components'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "isomap.nystrom":

        model = models.isomap.Nystrom(model_args, ratio=0.1, n_neighbors=model_args['#neighs'], n_components=model_args['#components'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "isomap.eng":

        model = models.isomap.ENG(model_args, n_neighbors=model_args['#neighs'], n_components=model_args['#components'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "isomap.adaptative":

        model = models.isomap.Adaptative(model_args, n_neighbors=model_args['#neighs'], n_components=model_args['#components'], k_max=20, eta=1e-3)
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "isomap.our":

        model = models.isomap.Our(model_args, model_args['#neighs'], model_args['#components'])
        Y = model.fit_transform(X)
        

    ########################################################
    # MVU ##################################################
    ########################################################


    elif model_args['model'].lower() == "mvu":

        model = models.mvu.MVU(model_args, model_args['#neighs'], model_args['eps'])
        Y = model.fit_transform(X)
    

    elif model_args['model'].lower() == "mvu.ineq":

        model = models.mvu.Ineq(model_args['#neighs'], model_args['eps'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "mvu.nystrom":

        model = models.mvu.Nystrom(model_args, model_args['#neighs'], model_args['eps'], 0.1)
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "mvu.eng":

        model = models.mvu.ENG(model_args, model_args['#neighs'], model_args['eps'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "mvu.adaptative":

        model = models.mvu.Adaptative(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "mvu.our":

        model = models.mvu.Our(model_args, model_args['#neighs'], model_args['eps'])
        Y = model.fit_transform(X)

    ########################################################
    # LE ###################################################
    ########################################################

    elif model_args['model'].lower() == "le":
        
        model = models.le.LaplacianEigenmaps(model_args, model_args['#neighs'], model_args['#components'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "le.skl":

        model = sklearn.manifold.SpectralEmbedding(n_neighbors=model_args['#neighs'], n_components=model_args['#components'])
        Y = model.fit_transform(X)

    ########################################################
    # LLE ##################################################
    ########################################################

    elif model_args["model"].lower() == "lle":
        
        model = models.lle.LocallyLinearEmbedding(model_args, model_args['#neighs'], model_args['#components'])
        Y = model.fit_transform(X)
        

    elif model_args["model"].lower() == "lle.skl":
        
        model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=model_args['#neighs'], n_components=model_args['#components'])
        Y = model.fit_transform(X)
        



    ########################################################
    # HLLE #################################################
    ########################################################

    elif model_args["model"].lower() == "hlle":
        
        model = models.hlle.HLLE(model_args, model_args['#neighs'], model_args['#components'])
        Y = model.fit_transform(X)

    elif model_args["model"].lower() == "hlle.skl":
    
        if model_args['#neighs'] <= model_args['#components'] * (model_args['#components'] + 3) / 2:
            # raise ValueError("n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
            print(f"Warning: n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
            return None
        
        model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=model_args['#neighs'], n_components=model_args['#components'], method='hessian', eigen_solver='dense')
        Y = model.fit_transform(X)

    ########################################################
    # T-SNE ################################################
    ########################################################
    
    
    elif model_args["model"].lower() == "tsne.skl":
        
        model = sklearn.manifold.TSNE(n_components=model_args['#components'])
        Y = model.fit_transform(X)
        
    
    ########################################################
    # LTSA #################################################
    ########################################################

    elif model_args["model"].lower() == "ltsa":
        
        model = models.ltsa.LTSA(model_args, model_args['#neighs'], model_args['#components'])
        Y = model.fit_transform(X)
        

    elif model_args["model"].lower() == "ltsa.skl":
        
        model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=model_args['#neighs'], n_components=model_args['#components'], method='ltsa', eigen_solver='dense')
        Y = model.fit_transform(X)
    
    ########################################################
    # UMAP #################################################
    ########################################################

    elif model_args["model"].lower() == "umap.lib":
        import umap  # Ensure the umap-learn library is installed
        model = umap.UMAP(n_neighbors=model_args['#neighs'], n_components=model_args['#components'])
        Y = model.fit_transform(X)
        

    ########################################################
    # END ##################################################
    ########################################################
    else:
        raise ValueError(f"Unknown model name {model_args['model']}.")

    # save_cache(model_args, Y, "Y")
    return Y
