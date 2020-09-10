"""From towardsdatascience.com:

Data ingestion such as retrieving 
data from CSV, relational database, NoSQL, Hadoop etc.
We have to retrieve data from multiple sources all the time
so we better to have a dedicated function for data retrieval.
"""
from glob import glob
import numpy as np
import os
import re
import soundfile as sf
import pickle

#TODO: create clean speech dataset, noise-aware dataset, hybrid dataset --> store in process dataset with metadata (SNR, etc.)

"""
Create datasets from input_data_dir
"""

def speech_list(input_data_dir):
    """
    Create clean speech

    Args:
        dataset_type (str, optional): [description]. Defaults to 'training'.

    Raises:
        ValueError: [description]

    Return:
        Audio_files (list)
    """
    
    data_dir = input_data_dir + 'EndtoEnd_VAD_data/'

    # List of files
    file_paths = glob(data_dir + '*.wav')

    # Remove input_data_dir from file_paths
    file_paths = [os.path.relpath(path, input_data_dir) for path in file_paths]

    return file_paths