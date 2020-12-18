import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import math

from python.dataset.ntcd_timit_dataset import speech_list, write_dataset
from python.processing.stft import stft
from python.processing.target import noise_robust_clean_speech_IBM, noise_robust_clean_speech_VAD
from python.utils import open_file


# Parameters
## Dataset
dataset_types = ['train', 'validation']

# dataset_size = 'subset'
dataset_size = 'complete'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')
output_pickle_dir = os.path.join('data', dataset_size, 'pickle/')

## STFT
video_frame_rate = 29.970030  # frames per second
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = math.floor((1 / (wlen_sec * video_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann' # type of window
center = False # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect' # This argument is ignored if center = False
pad_at_end = True # pad audio file at end to match same size after stft + istft
dtype = 'complex64'

## Noise robust VAD
vad_quantile_fraction_begin = 0.93
vad_quantile_fraction_end = 0.99
quantile_weight = 0.999

## Noise robust IBM
ibm_quantile_fraction = 0.999
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

                # Normalize audio
                speech = speech/(np.max(np.abs(speech)))

                if fs != fs_speech:
                    raise ValueError('Unexpected sampling rate')

                # TF reprepsentation
                speech_tf = stft(speech,
                                 fs=fs,
                                 wlen_sec=wlen_sec,
                                 win=win, 
                                 hop_percent=hop_percent,
                                 center=center,
                                 pad_mode=pad_mode,
                                 pad_at_end=pad_at_end,
                                 dtype=dtype) # shape = (freq_bins, frames)

                # binary mask
                speech_ibm = noise_robust_clean_speech_IBM(speech_tf,
                                                    vad_quantile_fraction_begin=vad_quantile_fraction_begin,
                                                    vad_quantile_fraction_end=vad_quantile_fraction_end,
                                                    ibm_quantile_fraction=ibm_quantile_fraction,
                                                    quantile_weight=quantile_weight)
                                    
                # # vad
                # speech_vad = noise_robust_clean_speech_VAD(speech_tf,
                #                                     quantile_fraction_begin=vad_quantile_fraction_begin,
                #                                     quantile_fraction_end=vad_quantile_fraction_end,
                #                                     quantile_weight=quantile_weight)
                
                if iteration == 0:
                    labels.append(speech_ibm)
                    # labels.append(speech_vad)

                if iteration == 1:
                    spectrograms.append(np.power(abs(speech_tf), 2))

            if iteration == 0:
                labels = np.concatenate(labels, axis=1)
                
                # write spectrograms
                write_dataset(labels,
                            output_data_dir=output_pickle_dir,
                            dataset_type=dataset_type,
                            suffix='labels')

                # # write spectrograms
                # write_dataset(labels,
                #             output_data_dir=output_pickle_dir,
                #             dataset_type=dataset_type,
                #             suffix='vad_labels')

                del labels            

            if iteration == 1:          
                spectrograms = np.concatenate(spectrograms, axis=1)
                # write spectrograms
                write_dataset(spectrograms,
                            output_data_dir=output_pickle_dir,
                            dataset_type=dataset_type,
                            suffix='frames')

                del spectrograms



            #open_file(output_pickle_dir)

if __name__ == '__main__':
    main()