import os
# import sys
import time
import json

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid

# sys.path.insert(1, "/rdmp_data/users/lourens/master_research/code/sdr/python") # add pySDR to system's path
from pySDR.SDR import SDR

from .metrics import DR_MetricsV1, LGC_Metrics

def save_results(results, outfile):
    """ Function to save optimization results to a JSON file.

    Parameters
    ----------
    results : dict
        A dictionary containing the results to be saved to `outfile`.
    outfile : str
        The name of the file the `results` should be saved to.
    """
    out_dict = {}
    if os.path.isfile(outfile):
        with open(outfile, 'r') as f:
            out_dict = json.load(f)

    for key in results.keys():
        out_dict[key] = results[key]

    with open(outfile, 'w') as f:
        json.dump(out_dict, f)

def optimize_DR(X, labels=None, num_samples=None, methods=["LMDS"], metric=None, storage_path="./", param_grid="./settings_DR.json", verbose=True, seed=None):
    """ Function that finds the optimal parameter set for each DR method given a parameter grid.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        An array containing the data that needs to be projected.
    labels : array-like, shape (n_samples,), default = None
        An array containing the labels (as numeric values) corresponding to each sample in X.
        Be sure to provide it when it is used by the optimization metric.
    num_samples : int, default = None (optional)
        Size of the random subset of samples that will be used to find the optimal DR parameters. If `None` all samples will be used.
        Beware that for large datasets this may significantly slow down the optimization procedure!
        As a general recommendation one should not use significantly more than 10000 samples.
    methods : list, default = ["LMDS"] (optional)
        A list with names of the DR methods to optimize as strings.
    metric : metrics.Metrics instance, default=None (optional)
        A `metrics.Metrics` instance with a `metric_total` method which will be called to evaluate the DR performance for a given parameter set.
        If not provided `metrics.DR_MetricsV1` will be initialized and used with its default parameters. 
    storage_path : str, default = "./" (optional)
        Path to the folder in which temporary files and results will be stored.
    param_grid : str, default = "./settings_DR.json" (optional)
        The path to a JSON file containing a *compact* parameter grid for *each* method provided in `methods`.
    verbose : bool, default = True (optional)
        Controls the verbosity.
    seed : int, default = None (optional)
        Random seed which is used by both the projection technique and for selecting a random subset of `num_samples`.

    Returns
    -------
    best_params : dict
        Dictionary containing the best parameter sets for each DR method specified in `methods`.
    best_scores : list
        List containing the best total scores for each DR method specified in `methods`. The scores are computed by calling `metric.metric_total`.
    """

    rng = np.random.default_rng(seed = seed) # initialize a random number generator

    # load parameter grids for different DR methods
    method_settings = {}
    if os.path.isfile(param_grid):
        with open(param_grid, 'r') as f:
            method_settings = json.load(f)
    else:
        raise ValueError("File {1} not found at {0}!".format(*os.path.split(param_grid)))

    # select random subset from the original dataset
    if num_samples is not None:
        keep_samples = rng.choice(X.shape[0], size=num_samples, replace=False)
        X = X[keep_samples]
        labels = labels[keep_samples]

    # if metric is not provided use DR_MetricsV1 with default parameters
    if metric is None:
        metric = DR_MetricsV1()

    # apply minmax normalization to X
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(X)

    # run DR for each method and grid point sequentially
    best_params = {}
    best_scores = {}
    for method in methods:
        method_path = os.path.join(storage_path, method)
        
        # make required directories
        if not os.path.isdir(method_path):
            os.mkdir(method_path)
        tmp_path = os.path.join(method_path, "tmp")
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path) # creat directory for temporary files

        lfs = open(os.path.join(method_path, "optimization_DR.log"), 'w') # log file stream

        # generate parameter grid from compact grid for the current method
        parameter_grid = list(ParameterGrid(method_settings[method]))

        best_scores[method] = 0
        reducer = SDR(path=tmp_path, data=X)
        for params in parameter_grid:
            print(f"Running {method} with {params}...")
            print(f"[{time.ctime()}] Running {method} with {params}...", file=lfs)

            # apply DR
            Y_DR = reducer.apply_DR(seed=seed, method=method, **params)

            # compute metric
            print(f"Computing metric...")
            print(f"[{time.ctime()}] Computing metric...", file=lfs)
            metric.fit(X, Y_DR, labels=labels)
            M = metric.metric_total()
            if verbose: print(f"\toptimization metric: {M}")
            print(f"[{time.ctime()}]", file=lfs)
            metric.print_summary(file=lfs)

            # store improved param set & score
            if M > best_scores[method]:
                best_params[method] = params
                best_scores[method] = M

            print("")

        del reducer

        if verbose: print(f"Results:\n\tBest total metric: {best_scores[method]}\n\tBest parameter set: {best_params[method]}\n")
        print(f"[{time.ctime()}]", file=lfs)
        print(f"Results:\n\tBest total metric: {best_scores[method]}\n\tBest parameter set: {best_params[method]}\n", file=lfs)

        lfs.close()

        # store best total metric in json file
        print("Saving results...")
        save_results(best_params, os.path.join(storage_path, "best_DR_params.json"))
        save_results(best_scores, os.path.join(storage_path, "best_DR_scores.json"))

    return best_params, best_scores

def optimize_LGC(X, labels=None, num_samples=None, methods=["LMDS"], metric=None, storage_path="./", param_grid="./settings_LGC.json", DR_params="./best_DR_params.json", verbose=True, seed=None):
    """ Function that finds the optimal parameter set for each LGC method given a parameter grid.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        An array containing the data that needs to be projected.
    labels : array-like, shape (n_samples,), default = None
        An array containing the labels (as numeric values) corresponding to each sample in X.
        Be sure to provide it when it is used by the optimization metric.
    num_samples : int, default = None (optional)
        Size of the random subset of samples that will be used to find the optimal LGC parameters. If `None` all samples will be used.
        Beware that for large datasets this may significantly slow down the optimization procedure!
        As a general recommendation one should not use significantly more than 10000 samples.
    methods : list, default = ["LMDS"] (optional)
        A list with names of the DR methods to use in combination with LGC as strings.
    metric : metrics.Metrics instance, default=None (optional)
        A `metrics.Metrics` instance with a `metric_total` method which will be called to evaluate the LGC performance for a given parameter set.
        If not provided `metrics.LGC_Metrics` will be initialized and used with its default parameters. 
    storage_path : str, default = "./" (optional)
        Path to the folder in which temporary files and results will be stored.
    param_grid : str, default = "./settings_LGC.json" (optional)
        The path to a JSON file containing a *compact* parameter grid for *each* method provided in `methods`.
    DR_params : str, default = "./best_DR_params.json" (optional)
        The path to a JSON file containing the parameters to use for *each* DR method provided in `methods`.
    verbose : bool, default = True (optional)
        Controls the verbosity.
    seed : int, default = None (optional)
        Random seed which is used by both the projection technique and for selecting a random subset of `num_samples`.

    Returns
    -------
    best_params : dict
        Dictionary containing the best LGC parameter set for each DR method used which were specified in `methods`.
    best_scores : list
        List containing the best total scores for each DR method used which were specified in `methods`. The scores are computed by calling `metric.metric_total`.
    """

    rng = np.random.default_rng(seed = seed) # initialize a random number generator

    # load LGC parameter grid
    LGC_settings = {}
    if os.path.isfile(param_grid):
        with open(param_grid, 'r') as f:
            LGC_settings = json.load(f)
    else:
        raise ValueError("File {1} not found at {0}!".format(*os.path.split(param_grid)))
    
    # generate parameter grid from compact grid
    parameter_grid = list(ParameterGrid(LGC_settings))

    # load DR settings for each DR method
    DR_settings = {}
    if os.path.isfile(DR_params):
        with open(DR_params, 'r') as f:
            DR_settings = json.load(f)
    else:
        raise ValueError("File {1} not found at {0}!".format(*os.path.split(DR_params)))

    # select random subset from the original dataset
    if num_samples is not None:
        keep_samples = rng.choice(X.shape[0], size=num_samples, replace=False)
        X = X[keep_samples]
        labels = labels[keep_samples]

    # if metric is not provided use LGC_Metrics with default parameters
    if metric is None:
        metric = LGC_Metrics()

    # apply minmax normalization to X
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(X)

    # run LGC & DR for each grid point sequentially
    best_params = {}
    best_scores = {}
    for method in methods:
        method_path = os.path.join(storage_path, method)
        
        # make required directories
        if not os.path.isdir(method_path):
            os.mkdir(method_path)
        tmp_path = os.path.join(method_path, "tmp")
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path) # creat directory for temporary files

        lfs = open(os.path.join(method_path, "optimization_LGC.log"), 'w') # log file stream

        best_scores[method] = 0
        for params in parameter_grid:
            print(f"Running S{method} with LGC parameters: {params} and DR parameters: {DR_settings[method]}...")
            print(f"[{time.ctime()}] Running S{method} with LGC parameters: {params} and DR parameters {DR_settings[method]}...", file=lfs)

            # apply SDR
            reducer = SDR(path=tmp_path, data=X)
            reducer.apply_LGC(**params)
            Y_SDR = reducer.apply_DR(seed=seed, method=method, **DR_settings[method])
            del reducer

            # compute metric
            print(f"Computing metric...")
            print(f"[{time.ctime()}] Computing metric...", file=lfs)
            metric.fit(X, Y_SDR, labels=labels)
            M = metric.metric_total()
            if verbose: print(f"\toptimization metric: {M}")
            print(f"[{time.ctime()}]", file=lfs)
            metric.print_summary(file=lfs)

            # store improved param set & score
            if M > best_scores[method]:
                best_params[method] = params
                best_scores[method] = M

            print("")

        if verbose: print(f"Results:\n\tBest total metric: {best_scores[method]}\n\tBest parameter set: {best_params[method]}\n")
        print(f"[{time.ctime()}]", file=lfs)
        print(f"Results:\n\tBest total metric: {best_scores[method]}\n\tBest parameter set: {best_params[method]}\n", file=lfs)

        lfs.close()

        # store best total metric in json file
        print("Saving results...")
        save_results(best_params, os.path.join(storage_path, "best_LGC_params.json"))
        save_results(best_scores, os.path.join(storage_path, "best_LGC_scores.json"))

    return best_params, best_scores