#!/usr/bin/env python3
import argparse
import configparser
import json
import subprocess
import multiprocessing
import time

if __name__ == "__main__":
    # command line argument definitions
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', help="configuration file")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="increase verbosity of the program")
    parser.add_argument('-V', '--versioning', action='store_true', 
                        help="Turns on versioning. Output files will be stored in the path specified by the storage path setting \
                            in the configuration file with a timestamp appended to it.")
    parser.add_argument('-M', '--multiprocessing', action='store_true',
                        help="Turns on multiprocessing. Each dimensionality reduction method will run on a separate thread.")
    args = parser.parse_args() # parse command line args

    if args.multiprocessing:
        print("Using multiprocessing!")

        # read configuration file
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(args.configuration)

        METHODS = json.loads(config['SDR_optimization_params']['methods'])

        # NB: versioning needs to be on at all times such that each subprocess has a separate working directory!
        # NB: verbosity is turned off since subprocesses do not print to the console anyways
        args = ["python3", "SHARC_pipeline.py", args.configuration, "--versioning"]

        # create suprocesses for each DR method
        processes = []
        # create tasks
        for method in METHODS:
            processes += [multiprocessing.Process(target=subprocess.call, args=(args+["--method", method],))]

        # start processes
        for process in processes:
            process.start()
            time.sleep(20) # wait 10 seconds before spawning next process

        # join processes
        for process in processes:
            process.join()
    else:
        args = ["python3", "SHARC_pipeline.py", args.configuration]
        if args.verbose: args += ["--verbose"]
        if args.versioning: args += ["--versioning"]
        subprocess.call(args)