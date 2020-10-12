import sys
sys.path.append('.')

import os
import pickle
import numpy as np
import time
import soundfile as sf
from tqdm import tqdm
import librosa
from sklearn.metrics import f1_score

from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
from python.models.spp_estimation import timo_mask_estimation
#from utils import count_parameters

from python.visualization import display_multiple_signals


# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

eps = 1e-8


# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'


## IBM
quantile_fraction = 0.98
quantile_weight = 0.999


## Timo's Classifier
model_name = 'classifier_timo'
model_data_dir = 'data/' + dataset_size + '/models/' + model_name + '/'

# Output_data_dir
output_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/')

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

def main():
    # Load input SNR
    all_snr_db = read_dataset(processed_data_dir, dataset_type, 'snr_db')
    all_snr_db = np.array(all_snr_db)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    for i, file_path in tqdm(enumerate(file_paths)):
        
        # Read files
        s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
        x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
        
        # x = x/np.max(x)
        T_orig = len(x_t)
        
        # TF representation
        # Input should be (frames, freq_bibs)
        x_tf = stft(x_t,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 dtype=dtype)
        
        # Power spectrogram (transpose)
        x = np.power(np.abs(x_tf), 2)

        # Estimate mask
        y_hat_soft = timo_mask_estimation(x)
        y_hat_hard = (y_hat_soft > 0.5).astype(int)

        # plots of target / estimation
        s_tf = stft(s_t,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 dtype=dtype) # shape = (freq_bins, frames) 

        # binary mask
        s_ibm = clean_speech_IBM(s_tf,
                                quantile_fraction=quantile_fraction,
                                quantile_weight=quantile_weight)

        # F1-score
        f1score_s_hat = f1_score(s_ibm.flatten(), y_hat_hard.flatten(), average="binary")

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
            [s_t, s_tf, s_ibm], # clean speech
            #[None, None, y_hat_hard]
            [None, None, y_hat_soft]
        ]
        #TODO: modify
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(all_snr_db[i])

        fig.suptitle(title, fontsize=40)
        
        # Save figure
        output_path = model_data_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        fig.savefig(output_path + '_soft_mask.png')

                ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
            [s_t, s_tf, s_ibm], # clean speech
            #[None, None, y_hat_hard]
            [None, None, y_hat_hard]
        ]
        #TODO: modify
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "F1-score = {:.3f} \n".format(all_snr_db[i], f1score_s_hat)

        fig.suptitle(title, fontsize=40)
        
        # Save figure
        output_path = model_data_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        fig.savefig(output_path + '_hard_mask.png')
        
if __name__ == '__main__':
    main()