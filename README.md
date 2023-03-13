SHARC
=====

Documentation
-------------
For an overview of the different functions included in the SHARC library and installation instructions please consult the [full documentation](https://martenlourens.github.io/SHARC/).

Using the SHARC Pipeline
------------------------
To make constructing and training an SDR-NNP classifier for a given dataset easier I wrote a program called `SHARC_pipeline.py`.
This program strings together the different functions in the SHARC library. To use this program please first install some additional requirements by running:

``` bash
$ pip install -r requirements.txt
```

For further information as to how this program works type:

``` bash
$ ./SHARC_pipeline.py --help
```

Please consult the `defaults.ini` as an example configuration file.