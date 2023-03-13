import os

def insertColors(table, colors):
    """ Combines magnitudes in astropy Table into colours and adds them to the table. The astropy Table is modified in-place.

    Parameters
    ----------
    table : astropy Table
        Table containing the magnitudes to be combined into colours.
    colors : iterable
        An array or list containing colours as strings with magnitudes corresponding to column names in the astropy Table.
    
    Returns
    ----------
    table : astropy Table
        Modified astropy Table with colours added as columns to the end of the original Table.
    
    """
    for color in colors:
        magnitudes = color.split("-")
        table.add_column(table[magnitudes[0]].data - table[magnitudes[1]].data, name=color)

def writeDataset(table, filename, verbose=True, overwrite=False):
    """ Writes an astropy Table to a fits file.

    Parameters
    ----------
    table : astropy Table
        Table to be written to file.
    filename : str
        Filename of the file the astropy Table needs to be written to.
    verbose : bool, default = True
        Variable controlling the verbosity of this function.
    overwrite : bool, default = False
        Variable controlling whether to overwrite any existing file. When the file already exists and `overwrite=False` the dataset won't be written and the function will exit.
    
    """
    if not os.path.isdir(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))

    if os.path.isfile(filename) and not overwrite:
        print(f"File {filename} already exists! To overwrite please set `overwrite=True` in the function call.")
        return

    if verbose: print(f"Writing data to {filename} ...")

    table.write(filename, format="fits", overwrite=overwrite)

    if verbose: print("Write successful!")