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
import pickle
from librosa.core import resample
from python.utils import get_key

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
    file_paths = glob(data_dir + '**/*.wav',recursive=True)

    # Remove input_speech_dir from file_paths
    file_paths = [os.path.relpath(path, input_speech_dir) for path in file_paths]

    return file_paths

"""
Noise-related
"""

def noise_list(input_noise_dir, dataset_type='test'):
    """[summary]

    Args:
        noise_dir ([type]): [description]
        dataset_type
    Return:
        subset_noise_paths
    """
    # List of files
    noise_paths = glob(input_noise_dir + '**/*.wav',recursive=True)

    # Remove input_noise_dir from noise_paths
    noise_paths = [os.path.relpath(noise_path, input_noise_dir) for noise_path in noise_paths]

    ### Training data
    if dataset_type == 'train':
        print('Not implemented')

    ### Validation data
    if dataset_type == 'validation':
        print('Not implemented')

    ### Test data
    if dataset_type == 'test':
        filenames = {
            'cafe': 'CAFE-CAFE-1.wav',
            'car': 'CAR-WINDOWNB-1.wav',
            'home': 'HOME-KITCHEN-1.wav',
            'street': 'STREET-CITY-1.wav'
        }        
        
        subset_noise_paths = {}

        # Subset of noise_paths matching filenames
        for noise_path in noise_paths:
            condition = any([filename in noise_path for filename in filenames.values()])
            if condition:
                subset_noise_paths[get_key(filenames, os.path.basename(noise_path))] = noise_path

    return subset_noise_paths

def preprocess_noise(noise_audio, key, fs_noise, fs):
    """[summary]

    Args:
        noise_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Read the 1st channel
    noise_audio = noise_audio[:,0]
    
    # Downsample to 16kHz
    if fs != fs_noise:
        noise_audio_resamp = resample(noise_audio, fs_noise, fs)

    # Trim begin/end of noise car
    if key == 'car':
        noise_audio_resamp = noise_audio_resamp[int(1.5*60*fs):int(43*60*fs)]

    return noise_audio_resamp

def noise_segment(noise_audios, noise_type, speech):
    """[summary]

    Args:
        noise_type ([type]): [description]
    """
    if noise_type in noise_audios.keys():
        noise_audio = noise_audios[noise_type]
        start = np.random.randint(len(noise_audio)-len(speech))
        noise_audio_seg = noise_audio[start:start+len(speech)]
    else:
        print('Error')
    return noise_audio_seg

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
    ### Training data
    if dataset_type == 'train':
        data_dir = 'si_tr_s'

    ### Validation data
    if dataset_type == 'validation':
        data_dir = 'si_dt_05'

    ### Test data
    if dataset_type == 'test':
        data_dir = 'si_et_05'
    
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

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