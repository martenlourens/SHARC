#!/usr/bin/env python3
import os
# import sys
import gc
import time
import argparse
import configparser
import json
from astropy.table import Table
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp

# sklearn imports
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# tensorflow imports
import tensorflow as tf

# XGBoost imports
from xgboost import XGBClassifier

# matplotlib imports
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# import pySDR
# sys.path.insert(1, "/rdmp_data/users/lourens/master_research/code/sdr/python") # add pySDR to system's path
from pySDR.SDR import SDR

# SHARC imports
from SHARC.utils import writeDataset
from SHARC.metrics import DR_MetricsV2, LGC_Metrics
from SHARC.optimization_routines import optimize_DR, optimize_LGC
from SHARC.nn_models import construct_NNPModel
from SHARC.nn_training_utils import train_nnp
from SHARC.classifiers import SDRNNPClassifier
from SHARC.plot_funcs import plot_projection, plot_projection_grid, plot_shepard_diagram, nnp_evaluation_plots, CustomConfusionMatrixDisplay

# global variables
CLASSIFIER_LUT = dict(
    KNN=KNeighborsClassifier,
    SVC=SVC,
    NNC=MLPClassifier,
    XGBC=XGBClassifier,
    DUMMY=DummyClassifier
    )

LOSS_FUNCTION = tf.keras.losses.MeanAbsoluteError()
OPTIMIZER = tf.keras.optimizers.Adam(1e-3)

if __name__ == "__main__":
    # command line argument definitions
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', help="configuration file")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="increase verbosity of the program")
    parser.add_argument('-V', '--versioning', action='store_true', 
                        help="Turns on versioning. Output files will be stored in the path specified by the storage path setting \
                            in the configuration file with a timestamp appended to it.")
    parser.add_argument('-m', '--method', help="DR method to use in SHARC pipeline. \
                        When not provided all DR methods specified in the configuration file will be used.")
    # parser.add_argument('-M', '--multiprocessing', action='store_true',
    #                     help="Turns on multiprocessing. Each dimensionality reduction method will run on a separate thread.")
    args = parser.parse_args() # parse command line args

    # read configuration file
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.configuration)

    if args.versioning:
        # add version information (i.e. a timestamp) to the storage path
        config['Paths']['storage_path'] = config['Paths']['storage_path'] + "_" + time.strftime('%Y%m%dT%H%M%S')

    if not os.path.isdir(config['Paths']['storage_path']):
        os.mkdir(config['Paths']['storage_path'])

    # load configuration parameters from config
    # globals
    RANDOM_SEED = int(config['Globals']['random_seed'])

    # SDR optimization parameters
    if args.method is not None:
        METHODS = [args.method]
    else:
        METHODS = json.loads(config['SDR_optimization_params']['methods'])
    NUM_SAMPLES = int(config['SDR_optimization_params']['num_samples'])
    NUM_NEIGHBORS = int(config['SDR_optimization_params']['num_neighbors'])
    DR_METRIC = DR_MetricsV2(k = NUM_NEIGHBORS)
    LGC_METRIC = LGC_Metrics(k = NUM_NEIGHBORS)

    # NNP model parameters
    NNP_VERSION = int(config['NNP_model_params']['version'])
    DROPOUT_RATE = float(config['NNP_model_params']['rate'])
    MOMENTUM = float(config['NNP_model_params']['momentum'])

    # NNP training parameters
    EPOCHS = int(config['NNP_training_params']['epochs'])
    NNP_TEST_SIZE = float(config['NNP_training_params']['test_size'])
    NNP_VALIDATION_SIZE = float(config['NNP_training_params']['validation_size'])

    # Classification parameters
    CLASSIFIERS = json.loads(config['Classification_params']['classifiers'])
    CLF_TEST_SIZE = float(config['Classification_params']['test_size'])

    # import dataset
    table = Table.read(config['Files']['dataset_file'], format="fits")
    with open(config['Files']['columns_file'], 'r') as f:
        columns = json.load(f)
    if args.verbose: print(f"label column: {columns['labels']}\nfeature columns: {columns['colors']}")

    # extract color data & label data from table
    X = np.ascontiguousarray(table[columns['colors']].to_pandas().values) # ensures X is C_CONTIGUOUS instead of F_CONTIGUOUS

    # apply minmax normalization to X
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(X)
    del scaler

    labels = table[[columns['labels']]].to_pandas().values.ravel()
    labels = labels.astype('int8')

    ######################
    ##  DR Optimization ##
    ######################
    print("Running DR optimization...")

    # find optimal DR parameters
    best_params, best_scores = optimize_DR(X, labels=labels, num_samples=NUM_SAMPLES, methods=METHODS,
                                        metric=DR_METRIC, storage_path=config['Paths']['storage_path'],
                                        param_grid=config['Files']['settings_DR_file'], verbose=args.verbose, 
                                        seed=RANDOM_SEED)
    
    # project full dataset using the best parameters for each DR method
    for method, params in best_params.items():
        method_path = os.path.join(config['Paths']['storage_path'], method)
        tmp_path = os.path.join(method_path, "tmp")
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)

        # apply DR
        reducer = SDR(path=tmp_path, data=X)
        Y_DR = reducer.apply_DR(seed=RANDOM_SEED, method=method, **params)
        del reducer

        # add to fits table
        for i, col in enumerate(Y_DR.T):
            name = f"Y_{method}{i+1}"
            # check if column already exists
            if name in table.colnames:
                table.replace_column(name=name, col=col)
            else:
                table.add_column(col=col, name=name)

        # plot projection
        ax = plot_projection(projection=Y_DR,
                            labels=labels)
        fig = ax.get_figure()
        fig.savefig(os.path.join(method_path, "projection_DR.png"), bbox_inches="tight", dpi=fig.get_dpi())
        plt.close(fig)

        # plot projection grid
        ax = plot_projection_grid(projection=Y_DR,
                                  labels=labels)
        fig = ax.get_figure()
        fig.savefig(os.path.join(method_path, "projection_grid_DR.png"), bbox_inches="tight", dpi=fig.get_dpi())
        plt.close(fig)

        # plot Shepard diagram (for random subset of dataset because of memory constraints)
        rng = np.random.default_rng(seed=RANDOM_SEED)
        keep_samples = rng.choice(X.shape[0], size=NUM_SAMPLES, replace=False)
        DR_METRIC.fit(X=X[keep_samples],
                      Y=Y_DR[keep_samples])
        MS, SD = DR_METRIC.metric_shepard_goodness(return_shepard=True)
        ax = plot_shepard_diagram(SD)
        title = ax.get_title()
        ax.set_title(title + " ($M_S = {:.6f}$)".format(MS))
        fig = ax.get_figure()
        fig.savefig(os.path.join(method_path, "projection_DR_Shepard_diagram.png"), bbox_inches="tight", dpi=fig.get_dpi())
        plt.close(fig)

        del SD
        del Y_DR

        gc.collect()
    
    # write projection data to FITS file
    writeDataset(table, 
                 filename=os.path.join(config['Paths']['storage_path'], os.path.split(config['Files']['dataset_file'])[-1]), 
                 verbose=args.verbose, overwrite=True)
    
    gc.collect()


    ##########################
    ##  LGC Optimization    ##
    ##########################
    print("Running LGC optimization...")
    DR_settings = best_params
    best_params, best_scores = optimize_LGC(X, labels=labels, num_samples=NUM_SAMPLES, methods=METHODS,
                                            metric=LGC_METRIC, storage_path=config['Paths']['storage_path'],
                                            param_grid=config['Files']['settings_LGC_file'],
                                            DR_params=os.path.join(config['Paths']['storage_path'], "best_DR_params.json"),
                                            verbose=args.verbose, seed=RANDOM_SEED)
    
    for method, params in best_params.items():
        method_path = os.path.join(config['Paths']['storage_path'], method)
        tmp_path = os.path.join(method_path, "tmp")
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)

        # apply SDR
        reducer = SDR(path=tmp_path, data=X)
        reducer.apply_LGC(**params)
        Y_SDR = reducer.apply_DR(seed=RANDOM_SEED, method=method, **DR_settings[method])
        del reducer

        # add to fits table
        for i, col in enumerate(Y_SDR.T):
            name = f"Y_S{method}{i+1}"
            # check if column already exists
            if name in table.colnames:
                table.replace_column(name=name, col=col)
            else:
                table.add_column(col=col, name=name)

        # plot projection
        ax = plot_projection(projection=Y_SDR,
                            labels=labels)
        fig = ax.get_figure()
        fig.savefig(os.path.join(method_path, "projection_SDR.png"), bbox_inches="tight", dpi=fig.get_dpi())
        plt.close(fig)

        # plot projection grid
        ax = plot_projection_grid(projection=Y_SDR,
                                labels=labels)
        fig = ax.get_figure()
        fig.savefig(os.path.join(method_path, "projection_grid_SDR.png"), bbox_inches="tight", dpi=fig.get_dpi())
        plt.close(fig)

        # plot Shepard diagram (for random subset of dataset because of memory constraints)
        rng = np.random.default_rng(seed=RANDOM_SEED)
        keep_samples = rng.choice(X.shape[0], size=NUM_SAMPLES, replace=False)
        LGC_METRIC.fit(X=X[keep_samples],
                       Y=Y_SDR[keep_samples])
        MS, SD = LGC_METRIC.metric_shepard_goodness(return_shepard=True)
        ax = plot_shepard_diagram(SD)
        title = ax.get_title()
        ax.set_title(title + " ($M_S = {:.6f}$)".format(MS))
        fig = ax.get_figure()
        fig.savefig(os.path.join(method_path, "projection_SDR_Shepard_diagram.png"), bbox_inches="tight", dpi=fig.get_dpi())
        plt.close(fig)

        del SD
        del Y_SDR

        gc.collect()

    # write projection data to FITS file
    writeDataset(table, 
                 filename=os.path.join(config['Paths']['storage_path'], os.path.split(config['Files']['dataset_file'])[-1]), 
                 verbose=args.verbose, overwrite=True)
    
    gc.collect()


    ######################
    ##  Train SDR-NNP   ##
    ######################
    print("Training SDR-NNP models...")
    for method in METHODS:
        method_path = os.path.join(config['Paths']['storage_path'], method)

        print(f"Training SDR-NNP model for S{method}...")
        Y = np.ascontiguousarray(table[[f"Y_S{method}1", f"Y_S{method}2"]].to_pandas().values)
        
        # preprocess output features such that values lie between 0 and 1
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
        scaler.fit_transform(Y)
        del scaler

        # split data into training and test sets
        X_train, X_test, Y_train, Y_test, labels_train, labels_test = train_test_split(X, Y, labels,
                                                                                       test_size=NNP_TEST_SIZE, 
                                                                                       shuffle=True, 
                                                                                       stratify=labels, 
                                                                                       random_state=RANDOM_SEED)
        
        if args.verbose: print(f"Train samples: {len(labels_train)}\nTest samples: {len(labels_test)}")

        # construct neural network
        model = construct_NNPModel(X_train.shape[1], output_dimensions=2, output_activation="sigmoid", 
                                   version=NNP_VERSION, D1_units=X_train.shape[1], rate=DROPOUT_RATE, momentum=MOMENTUM)
        
        if args.verbose: print(model.summary())

        # train neural network
        train_loss, valid_loss, pred_train_loss = train_nnp(X_train, Y_train, model, 
                                                            LOSS_FUNCTION, OPTIMIZER, 
                                                            labels=labels_train, epochs=EPOCHS, 
                                                            validation_ratio=NNP_VALIDATION_SIZE, 
                                                            save_path=method_path,
                                                            verbose=args.verbose)
        
        # NN EVALUATION:
        # project test set
        Y_pred = model(X_test, training=False)
        fig = nnp_evaluation_plots(Y_test, Y_pred, 
                                  train_loss=train_loss, 
                                  pred_train_loss=pred_train_loss, 
                                  valid_loss=valid_loss,
                                  loss_function=LOSS_FUNCTION,
                                  labels=labels_test)
        fig.savefig(os.path.join(method_path, model.name, "SDR-NNP_test.png"), bbox_inches="tight", dpi=fig.dpi)
        plt.close(fig)
        del Y_pred

        # project full set
        Y_pred = model(X, training=False)
        fig = nnp_evaluation_plots(Y, Y_pred, 
                                  train_loss=train_loss, 
                                  pred_train_loss=pred_train_loss, 
                                  valid_loss=valid_loss,
                                  loss_function=LOSS_FUNCTION,
                                  labels=labels)
        fig.savefig(os.path.join(method_path, model.name, "SDR-NNP_full.png"), bbox_inches="tight", dpi=fig.dpi)
        plt.close(fig)
        del Y_pred

        del Y
        del X_train
        del X_test
        del Y_train
        del Y_test
        del labels_train
        del labels_test
        del train_loss
        del valid_loss
        del pred_train_loss

        gc.collect()

    gc.collect()

    #################################
    ##  Train & Test Classifiers   ##
    #################################
    display_labels = ["STAR", "GAL", "QSO"]
    df = pd.DataFrame(columns=["Precision", "Recall", "F1 Score", "Class", "DR Technique", "Classifier"]) # create empty dataframe with column names

    print("Training classifiers...")
    for method in METHODS:
        method_path = os.path.join(config['Paths']['storage_path'], method)
        model_path = os.path.join(method_path, model.name)
        classifier_path = os.path.join(method_path, "Classifiers")
        if not os.path.isdir(classifier_path):
            os.mkdir(classifier_path)

        # split data into training and test sets
        X_train, X_test, labels_train, labels_test = train_test_split(X, labels,
                                                                      test_size=CLF_TEST_SIZE, 
                                                                      shuffle=True, 
                                                                      stratify=labels, 
                                                                      random_state=RANDOM_SEED)

        for classifier in CLASSIFIERS:
            print(f"Training S{method}-NNP {classifier} classifier...")
            clf = SDRNNPClassifier(
                nnp_model_path=model_path, 
                classifier=CLASSIFIER_LUT[classifier.split("_")[0]](**json.loads(config["Classification_params"][classifier.lower() + "_kwargs"]))
                )
            clf.fit(X_train, labels_train)

            # evaluate performance
            labels_pred = clf.predict(X_test)
            accuracy = accuracy_score(labels_test, labels_pred)
            precision = precision_score(labels_test, labels_pred, average=None)
            recall = recall_score(labels_test, labels_pred, average=None)
            f1 = f1_score(labels_test, labels_pred, average=None)

            # add scores to pandas dataframe
            data = np.stack((precision, recall, f1, display_labels, 3 * [f"S{method}-NNP"], 3 * [classifier]), axis=-1)
            df = pd.concat((df, pd.DataFrame(data=data, columns=df.columns)))

            # make plots to visualize performance
            ncols = 2
            if classifier.split("_")[0] == "NNC":
                height = 10
                nrows = 2
            else:
                height = 5
                nrows = 1

            fig = plt.figure()
            fig.set_size_inches(w=10, h=height)
            gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
            axs = [fig.add_subplot(gs[0,i]) for i in range(ncols)]

            # plot domains
            axs[0].set_aspect("equal")
            axs[0].set_title(f"Decision Boundaries and True Labels\n test accuracy: {accuracy:.6f}")
            clf.plot_classifier_decision_boundaries(ax=axs[0])
            clf.plot_projection(X_test, y=labels_test, ax=axs[0])

            # plot confusion matrix
            axs[1].set_title("Confusion Matrix Normalized over Predicted Labels", pad=25)
            disp_CM = CustomConfusionMatrixDisplay.from_predictions_with_counts(labels_test, labels_pred, normalize="pred", ax=axs[1])

            if classifier.split("_")[0] == "NNC":
                axs += [fig.add_subplot(gs[1,:])]
                
                # plot loss curve
                axs[2].plot(range(1, clf.classifier.n_iter_+1), clf.classifier.loss_curve_, label="Training loss")
                axs[2].plot(range(1, clf.classifier.n_iter_+1), clf.classifier.validation_scores_, label="Validation accuracy")
                axs[2].set_xlabel("Epochs")
                axs[2].set_xlim((1,None))
                axs[2].legend()
                axs[2].grid()

            fig.tight_layout()
            fig.savefig(os.path.join(classifier_path, f"S{method}-NNP_{classifier}_evaluation.png"), bbox_inches="tight", dpi=fig.dpi)
            plt.close(fig)
            
            # serialize trained classifier using pickle
            pkl_fname = os.path.join(classifier_path, f"S{method}-NNP_{classifier}.pkl")
            if args.verbose: print(f"Writing S{method}-NNP {classifier} classifier to {pkl_fname}...")
            with open(pkl_fname, 'wb') as f:
                pickle.dump(clf, f)

            del labels_pred
            del data
            del clf

            gc.collect()

        del X_train
        del X_test
        del labels_train
        del labels_test

        gc.collect()

    # save classification scores to *.csv
    cs_fname = os.path.join(config['Paths']['storage_path'], "classification_scores.csv")
    if args.verbose: print(f"Writing classification scores to {cs_fname}")
    df.to_csv(cs_fname, index=False)

    del df

    gc.collect()