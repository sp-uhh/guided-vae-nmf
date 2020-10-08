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
import collections
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

        folder_names = {
            'domestic': 'DWASHING',
            'nature': 'NRIVER',
            'office': 'OOFFICE',
            'transportation': 'TMETRO'
        }   

    ### Validation data
    if dataset_type == 'validation':

        folder_names = {
            'nature': 'NFIELD',
            'office': 'OHALLWAY',
            'public': 'PSTATION',
            'transportation': 'TBUS'
        }   

    ### Test data
    if dataset_type == 'test':
        print('Not implemented')
     

    subset_noise_paths = collections.defaultdict(dict)

    # Subset of noise_paths matching folder_names
    for noise_path in noise_paths:
        condition = any([folder_name in noise_path for folder_name in folder_names.values()])
        if condition:
            sample_id = int(''.join(filter(str.isdigit, noise_path)))
            subset_noise_paths[get_key(folder_names, os.path.dirname(noise_path))][sample_id] = noise_path

    return subset_noise_paths

def preprocess_noise(noise_audio, fs_noise, fs):
    """[summary]

    Args:
        noise_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Downsample to 16kHz
    if fs != fs_noise:
        noise_audio_resamp = resample(noise_audio, fs_noise, fs)

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