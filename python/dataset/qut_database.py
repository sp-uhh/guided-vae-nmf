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
from pathlib import Path
from librosa.core import resample # do not use it because slow, use pysox instead
from python.utils import get_key

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
    #TODO: modify test set
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
        noise_audio_resamp = noise_audio_resamp[int(1.5*60*fs):int(43*60*fs)] # Extract part between 1.5min and 43min

    return noise_audio_resamp

def noise_list_preprocessed(preprocessed_noise_dir, dataset_type='test'):
    """[summary]

    Args:
        noise_dir ([type]): [description]
        dataset_type
    Return:
        subset_noise_paths
    """
    data_dir = preprocessed_noise_dir

    ### Training data
    if dataset_type == 'train':
        print('Not implemented')

    ### Validation data
    if dataset_type == 'validation':
        print('Not implemented')

    ### Test data
    if dataset_type == 'test':
        data_dir += 'si_et_05/'

    # List of files
    noise_paths = glob(data_dir + '**/*.wav',recursive=True)

    # Store as a dict
    noise_paths = {Path(i).stem:i for i in noise_paths}
    return noise_paths

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