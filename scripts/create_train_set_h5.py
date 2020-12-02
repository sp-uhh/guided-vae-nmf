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

input_speech_dir = os.path.join('data', dataset_size, 'raw/')
output_data_dir = os.path.join('data', dataset_size, 'h5/')
data_dir = 'CSR-1-WSJ-0'

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.98
quantile_weight = 0.999

def main():

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    output_h5_dir = output_data_dir + data_dir + '.h5'

    #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
    with h5.File(output_h5_dir, 'w', rdcc_nbytes=1024**2*400, rdcc_nslots=10e6) as f:
        
        for dataset_type in dataset_types:

            # Exact shape of dataset is unknown in advance unfortunately
            # Faster writing if you know the shape in advance
            # Size of chunks corresponds to one spectrogram frame
            f.create_dataset('X_' + dataset_type, shape=(513, 0), dtype=np.float32, maxshape=(513, None), chunks=(513, 1), compression="lzf")
            f.create_dataset('Y_' + dataset_type, shape=(513, 0), dtype=np.float32, maxshape=(513, None), chunks=(513, 1), compression="lzf")
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