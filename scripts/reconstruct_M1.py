import sys
sys.path.append('.')

import os
import pickle
import numpy as np
import torch
from torch import nn
import time
import soundfile as sf
from tqdm import tqdm
import librosa

from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset
from python.processing.stft import stft
from python.models.models import VariationalAutoencoder
#from utils import count_parameters

from python.visualization import display_multiple_spectro


# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

cuda = torch.cuda.is_available()
eps = 1e-8


# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'



## Deep Generative Model
model_name = 'M1_end_epoch_050/M1_epoch_036_vloss_465.28'
x_dim = 513 # frequency bins (spectrogram)
z_dim = 128
h_dim = [256, 128]

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

    device = torch.device("cuda" if cuda else "cpu")
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    model = VariationalAutoencoder([x_dim, z_dim, h_dim])
    model.load_state_dict(torch.load(os.path.join('models', model_name + '.pt')))
    if cuda: model = model.cuda()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

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
                        
        # Transpose to match PyTorch
        x_tf = x_tf.T # (frames, freq_bins)
        
        # Power spectrogram (transpose)
        x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)

        # Encode-decode
        reconstruction = model(x)
        reconstruction = reconstruction.cpu().numpy()

        # plots of target / estimation
        s_tf = stft(s_t,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 dtype=dtype) # shape = (freq_bins, frames) 

        # Transpose to match librosa.display
        reconstruction = reconstruction.T

        # Transform to dB
        x_psd = x.cpu().numpy().T
        x_psd = librosa.core.power_to_db(x_psd)          

        s_psd = np.power(abs(s_tf),2)
        s_psd = librosa.core.power_to_db(s_psd)          

        reconstruction = librosa.core.power_to_db(reconstruction)          

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_psd], # mixture: (waveform, tf_signal, no mask)
            [s_t, s_psd], # clean speech
            [None, reconstruction]
        ]
        #TODO: modify
        fig = display_multiple_spectro(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "".format(all_snr_db[i])

        fig.suptitle(title, fontsize=40)

        # Save figure
        fig.savefig(model_data_dir + os.path.splitext(file_path)[0] + '_recon.png')
        
if __name__ == '__main__':
    main()