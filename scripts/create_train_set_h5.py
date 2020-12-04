import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import tables #needed for blosc compression
import h5py as h5

from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM, clean_speech_VAD
from python.utils import open_file


# Parameters
## Dataset
dataset_types = ['train', 'validation']
# dataset_types = ['train']
# dataset_types = ['validation']

# dataset_size = 'subset'
dataset_size = 'complete'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')
output_data_dir = os.path.join('data', dataset_size, 'h5/')
data_dir = 'CSR-1-WSJ-0'
# suffix = 'lzf'
# suffix = 'lzf_conda'
suffix = 'lzf_conda_bis'
# suffix = 'lzf_pip'
# suffix = 'lzf_shuffle_ter'
# suffix = 'blosc_nslots1e5'
# suffix = 'blosc_importafter'
# suffix = 'blosc_conda'

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.98
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
chunks = (513, 1)
# chunks = None
compression = 'lzf'
# compression = 32001
# compression = None
# shuffle = True
shuffle = None

def main():

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    output_h5_dir = output_data_dir + data_dir + '_' + suffix + '.h5'

    for dataset_type in dataset_types:
    
        with h5.File(output_h5_dir, 'a', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    

            # Delete datasets if already exists
            #TODO: change way of suppressing dataset
            if 'X_' + dataset_type in f:
                del f['X_' + dataset_type]
                del f['Y_' + dataset_type]
            
            # Exact shape of dataset is unknown in advance unfortunately
            # Faster writing if you know the shape in advance
            # Size of chunks corresponds to one spectrogram frame
            f.create_dataset('X_' + dataset_type, shape=(513, 0), dtype=np.float32, maxshape=(513, None), chunks=chunks, compression=compression, shuffle=shuffle)
            f.create_dataset('Y_' + dataset_type, shape=(513, 0), dtype=np.float32, maxshape=(513, None), chunks=chunks, compression=compression, shuffle=shuffle)
            f.attrs['fs'] = fs
            f.attrs['wlen_sec'] = wlen_sec
            f.attrs['hop_percent'] = hop_percent
            f.attrs['win'] = win
            f.attrs['dtype'] = dtype

            f.attrs['quantile_fraction'] = quantile_fraction
            f.attrs['quantile_weight'] = quantile_weight

            # Create file list
            file_paths = speech_list(input_speech_dir=input_speech_dir,
                                    dataset_type=dataset_type)

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

                # binary mask
                speech_ibm = clean_speech_IBM(speech_tf,
                                        quantile_fraction=quantile_fraction,
                                        quantile_weight=quantile_weight)
                                    
                # # vad
                # speech_vad = clean_speech_VAD(speech_tf,
                #                         quantile_fraction=quantile_fraction,
                #                         quantile_weight=quantile_weight)
                
                spectrogram = np.power(abs(speech_tf), 2)

                f['X_' + dataset_type].resize((f['X_' + dataset_type].shape[1] + spectrogram.shape[1]), axis = 1)
                f['X_' + dataset_type][:,-spectrogram.shape[1]:] = spectrogram

                label = speech_ibm
                # labels.append(speech_vad)

                f['Y_' + dataset_type].resize((f['Y_' + dataset_type].shape[1] + label.shape[1]), axis = 1)
                f['Y_' + dataset_type][:,-label.shape[1]:] = label

if __name__ == '__main__':
    main()