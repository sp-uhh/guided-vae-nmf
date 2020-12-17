import sys
sys.path.append('.')

from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset, read_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
import soundfile as sf
import numpy as np
from numpy.testing import assert_array_equal
from python.utils import open_file


# Parameters
## Dataset
input_speech_dir = 'data/subset/raw/'
output_data_dir = 'data/subset/pickle/'
dataset_type = 'test'
fs = int(16e3) # Sampling rate

## STFT
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window

## Ideal binary mask
quantile_fraction = 0.98
quantile_weight = 0.999
#eps = np.finfo(float).eps # machine epsilon

def main():

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    spectrograms = []
    labels = []

    for path in file_paths:
        #paths.append(path)
        x, fs_x = sf.read(input_speech_dir + path, samplerate=None)
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

        spectrograms.append(np.power(np.abs(x_tf), 2))
        labels.append(x_ibm)

    spectrograms = np.concatenate(spectrograms, axis=1)
    labels = np.concatenate(labels, axis=1)

    # write spectrograms + labels
    write_dataset(spectrograms,
                  output_data_dir=output_data_dir,
                  dataset_type=dataset_type,
                  suffix='frames')
    
    write_dataset(labels,
                output_data_dir=output_data_dir,
                dataset_type=dataset_type,
                suffix='labels')


    # Read pickle
    pickle_spectrograms = read_dataset(data_dir=output_data_dir,
                 dataset_type=dataset_type,
                 suffix='frames')
    
    pickle_labels = read_dataset(data_dir=output_data_dir,
                dataset_type=dataset_type,
                suffix='labels') 
    
    # Assert stored data is same
    assert_array_equal(spectrograms, pickle_spectrograms)
    assert_array_equal(labels, pickle_labels)

    #Open output directory
    open_file(output_data_dir)
    
if __name__ == '__main__':
    main()