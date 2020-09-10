from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset, read_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
import numpy as np
import soundfile as sf
from numpy.testing import assert_array_equal
import pytest

@pytest.mark.parametrize('dataset_type',
[
    ('train'),
    ('validation'),
    ('test')
]
)
def test_write_read_frames(dataset_type):
    # Parameters
    ## Dataset
    input_data_dir = 'data/subset/raw/'
    output_data_dir = 'data/subset/processed/'
    fs = int(16e3) # Sampling rate

    ## STFT
    wlen_sec = 64e-3 # window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    win = 'hann' # type of window

    # Create file list
    file_paths = speech_list(input_data_dir=input_data_dir,
                             dataset_type=dataset_type)

    spectrograms = []

    for path in file_paths:

        x, fs_x = sf.read(input_data_dir + path, samplerate=None)
        x = x/np.max(np.abs(x))
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent)

        spectrograms.append(np.power(np.abs(x_tf), 2))

    spectrograms = np.concatenate(spectrograms, axis=0)

    # write spectrograms
    write_dataset(spectrograms,
                  output_data_dir=output_data_dir,
                  dataset_type=dataset_type,
                  suffix='frames')
    
    # Read pickle
    pickle_spectrograms = read_dataset(data_dir=output_data_dir,
                 dataset_type=dataset_type,
                 suffix='frames') 
    
    # Assert stored data is same as spectrograms
    assert_array_equal(spectrograms, pickle_spectrograms)

@pytest.mark.parametrize('dataset_type',
[
    ('train'),
    ('validation'),
    ('test')
]
)
def test_write_read_labels(dataset_type):
    # Parameters
    ## Dataset
    input_data_dir = 'data/subset/raw/'
    output_data_dir = 'data/subset/processed/'
    fs = int(16e3) # Sampling rate

    ## STFT
    wlen_sec = 64e-3 # window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    win = 'hann' # type of window

    ## Ideal binary mask
    quantile_fraction = 0.98
    quantile_weight = 0.999

    # Create file list
    file_paths = speech_list(input_data_dir=input_data_dir,
                             dataset_type=dataset_type)

    labels = []

    for path in file_paths:

        x, fs_x = sf.read(input_data_dir + path, samplerate=None)
        x = x/np.max(np.abs(x))
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent)

         # binary mask
        x_ibm = clean_speech_IBM(x_tf,
                                 quantile_fraction=quantile_fraction,
                                 quantile_weight=quantile_weight)

        labels.append(x_ibm)

    labels = np.concatenate(labels, axis=0)

    # write spectrograms
    write_dataset(labels,
                  output_data_dir=output_data_dir,
                  dataset_type=dataset_type,
                  suffix='labels')
    
    # Read pickle
    pickle_labels = read_dataset(data_dir=output_data_dir,
                 dataset_type=dataset_type,
                 suffix='labels') 
    
    # Assert stored data is same as spectrograms
    assert_array_equal(labels, pickle_labels)