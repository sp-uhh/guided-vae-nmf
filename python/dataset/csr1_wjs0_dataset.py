"""From towardsdatascience.com:

Data ingestion such as retrieving 
data from CSV, relational database, NoSQL, Hadoop etc.
We have to retrieve data from multiple sources all the time
so we better to have a dedicated function for data retrieval.
"""
from glob import glob
import os
import pickle
import h5py

#TODO: create clean speech dataset, noise-aware dataset, hybrid dataset --> store in process dataset with metadata (SNR, etc.)

"""
Speech-related
"""

def speech_list(input_speech_dir,
                dataset_type='train'):
    """
    Create clean speech + clean speech VAD

    Args:
        dataset_type (str, optional): [description]. Defaults to 'training'.

    Raises:
        ValueError: [description]

    Return:
        Audio_files (list)
    """
    
    data_dir = input_speech_dir + 'CSR-1-WSJ-0/WAV/wsj0/'

    ### Training data
    if dataset_type == 'train':
        data_dir += 'si_tr_s/'

    ### Validation data
    if dataset_type == 'validation':
        data_dir += 'si_dt_05/'

    ### Test data
    if dataset_type == 'test':
        data_dir += 'si_et_05/'

    # List of files
    file_paths = sorted(glob(data_dir + '**/*.wav',recursive=True))

    # Remove input_speech_dir from file_paths
    file_paths = [os.path.relpath(path, input_speech_dir) for path in file_paths]

    return file_paths
"""
Read/write pickles
"""

def write_dataset(data,
                  output_data_dir,
                  dataset_type,
                  suffix='unlabeled_frames'):
    """
    Store data in pickle file

    Args:
        dataset ([type]): [description]
        ouput_data_dir ([type]): [description]
        dataset_type ([type]): [description]
        suffix (str, optional): [description]. Defaults to 'frames'.
    """

    data_dir = 'CSR-1-WSJ-0/'
        
    if not os.path.exists(output_data_dir + data_dir):
        os.makedirs(output_data_dir + data_dir)

    ### Training data
    if dataset_type == 'train':
        data_dir += 'si_tr_s'

    ### Validation data
    if dataset_type == 'validation':
        data_dir += 'si_dt_05'

    ### Test data
    if dataset_type == 'test':
        data_dir += 'si_et_05'

    output_data_path = output_data_dir + data_dir + '_' + suffix + '.p'

    with open(output_data_path, 'wb') as file:
        pickle.dump(data, file, protocol=4)
    
    print("data is stored in " + output_data_dir)


def read_dataset(data_dir,
                 dataset_type,
                 suffix='unlabeled_frames'):
    """
    Store data in pickle file

    Args:
        dataset ([type]): [description]
        ouput_data_dir ([type]): [description]
        dataset_type ([type]): [description]
        suffix (str, optional): [description]. Defaults to 'frames'.
    """

    data_dir += 'CSR-1-WSJ-0/'

    ### Training data
    if dataset_type == 'train':
        data_dir += 'si_tr_s'

    ### Validation data
    if dataset_type == 'validation':
        data_dir += 'si_dt_05'

    ### Test data
    if dataset_type == 'test':
        data_dir += 'si_et_05'

    data_dir += '_' + suffix + '.p'

    with open(data_dir, 'rb') as data_path:
        data = pickle.load(data_path)
    return data