import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm

from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM, clean_speech_VAD
from python.utils import open_file


# Parameters
## Dataset
dataset_types = ['train', 'validation']

# dataset_size = 'subset'
dataset_size = 'complete'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')

input_noise_dir = 'data/complete/raw/qutnoise_databases/' # change the name of the subfolder in your computer
output_noise_dir = 'data/complete/processed/qutnoise_databases/' # change the name of the subfolder in your computer

output_pickle_dir = os.path.join('data', dataset_size, 'pickle/')

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

    for dataset_type in dataset_types:

        # Create file list
        file_paths = speech_list(input_speech_dir=input_speech_dir,
                                dataset_type=dataset_type)

        spectrograms = []
        labels = []

        # Do 2 iterations to save separately spectro and labels (RAM issues)
        for iteration in range(2):

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

                # # binary mask
                # speech_ibm = clean_speech_IBM(speech_tf,
                #                         quantile_fraction=quantile_fraction,
                #                         quantile_weight=quantile_weight)
                                    
                # vad
                speech_vad = clean_speech_VAD(speech_tf,
                                        quantile_fraction=quantile_fraction,
                                        quantile_weight=quantile_weight)
                
                if iteration == 0:
                    # labels.append(speech_ibm)
                    labels.append(speech_vad)

                # if iteration == 1:
                #     spectrograms.append(np.power(abs(speech_tf), 2))

            if iteration == 0:
                labels = np.concatenate(labels, axis=1)
                
                # write spectrograms
                write_dataset(labels,
                            output_data_dir=output_pickle_dir,
                            dataset_type=dataset_type,
                            suffix='vad_labels')

                del labels            

            # if iteration == 1:          
            #     spectrograms = np.concatenate(spectrograms, axis=1)
            #     # write spectrograms
            #     write_dataset(spectrograms,
            #                 output_data_dir=output_pickle_dir,
            #                 dataset_type=dataset_type,
            #                 suffix='frames')

            #     del spectrograms



            #open_file(output_pickle_dir)

if __name__ == '__main__':
    main()