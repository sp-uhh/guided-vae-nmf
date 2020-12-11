import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import h5py as h5

from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM, clean_speech_VAD
from python.utils import open_file

# Parameters
## Dataset
dataset_types = ['train', 'validation']

dataset_size = 'subset'
# dataset_size = 'complete'

# data_dir = 'h5'
data_dir = 'export'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')
output_dataset_dir = os.path.join('data', dataset_size, data_dir + '/')
dataset_name = 'CSR-1-WSJ-0'
labels = 'labels'
# labels = 'vad_labels'

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.999
quantile_weight = 0.999

# HDF5 parameters
rdcc_nbytes = 1024**2*400 # The number of bytes to use for the chunk cache
                          # Default is 1 Mb
                          # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)
                  # for compression 'zlf' --> 1e4 - 1e7
                  # for compression 32001 --> 1e4
X_shape = (513, 0)
X_maxshape = (513, None)
X_chunks = (513, 1)

if labels == 'labels':
    Y_shape = (513, 0)
    Y_maxshape = (513, None)
    Y_chunks = (513, 1)
    
if labels == 'vad_labels':
    Y_shape = (1, 0)
    Y_maxshape = (1, None)    
    Y_chunks = (1,1)    
    
compression = 'lzf'
shuffle = False

def main():

    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    output_h5_dir = output_dataset_dir + dataset_name + '_' + labels + '.h5'

    with h5.File(output_h5_dir, 'a', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    

        for dataset_type in dataset_types:
    
            # Delete datasets if already exists
            if 'X_' + dataset_type in f:
                del f['X_' + dataset_type]
                del f['Y_' + dataset_type]
            
            # Exact shape of dataset is unknown in advance unfortunately
            # Faster writing if you know the shape in advance
            # Size of chunks corresponds to one spectrogram frame
            f.create_dataset('X_' + dataset_type, shape=X_shape, dtype='float32', maxshape=X_maxshape, chunks=X_chunks, compression=compression, shuffle=shuffle)
            f.create_dataset('Y_' + dataset_type, shape=Y_shape, dtype='float32', maxshape=Y_maxshape, chunks=Y_chunks, compression=compression, shuffle=shuffle)
            
            # STFT attributes
            f.attrs['fs'] = fs
            f.attrs['wlen_sec'] = wlen_sec
            f.attrs['hop_percent'] = hop_percent
            f.attrs['win'] = win
            f.attrs['dtype'] = dtype

            # label attributes
            f.attrs['quantile_fraction'] = quantile_fraction
            f.attrs['quantile_weight'] = quantile_weight

            # HDF5 attributes
            f.attrs['X_chunks'] = X_chunks
            f.attrs['Y_chunks'] = Y_chunks
            f.attrs['compression'] = compression

            # Create file list
            file_paths = speech_list(input_speech_dir=input_speech_dir,
                                    dataset_type=dataset_type)

            # Store dataset in variables for faster I/O
            fx = f['X_' + dataset_type]
            fy = f['Y_' + dataset_type]

            for i, file_path in tqdm(enumerate(file_paths)):

                speech, fs_speech = sf.read(input_speech_dir + file_path, samplerate=None)

                # Cut burst at begining of file
                speech = speech[int(0.1*fs):]

                # Normalize audio
                speech = speech/(np.max(np.abs(speech)))

                if fs != fs_speech:
                    raise ValueError('Unexpected sampling rate')

                # TF reprepsentation
                speech_tf = stft(speech, fs=fs, wlen_sec=wlen_sec, win=win, 
                    hop_percent=hop_percent, dtype=dtype)
                
                spectrogram = np.power(abs(speech_tf), 2)
                
                if labels == 'vad_labels':             
                    # vad
                    speech_vad = clean_speech_VAD(speech_tf,
                                            quantile_fraction=quantile_fraction,
                                            quantile_weight=quantile_weight)

                    label = speech_vad

                if labels == 'labels':
                    # binary mask
                    speech_ibm = clean_speech_IBM(speech_tf,
                                            quantile_fraction=quantile_fraction,
                                            quantile_weight=quantile_weight)

                    label = speech_ibm
                    
                # Store spectrogram in dataset
                fx.resize((fx.shape[1] + spectrogram.shape[1]), axis = 1)
                fx[:,-spectrogram.shape[1]:] = spectrogram

                # Store spectrogram in label
                fy.resize((fy.shape[1] + label.shape[1]), axis = 1)
                fy[:,-label.shape[1]:] = label

if __name__ == '__main__':
    main()