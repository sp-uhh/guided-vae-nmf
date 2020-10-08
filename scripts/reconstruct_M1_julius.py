import sys
sys.path.append('.')

import os
import pickle
import numpy as np
import torch
from torch import nn
from librosa import stft, istft
from tqdm import tqdm
import soundfile as sf
import librosa

from python.models.vae import VAE
from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset

from python.visualization import display_multiple_spectro

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = open('output.log','w') 

print('Torch version: {}'.format(torch.__version__))
print('Device: %s' % (device))
if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

# test_data = pickle.load(open('../data/mixture.p', 'rb'))

model_name = 'vae/vae_200_vloss_0476'
model_path = 'models/vae/vae_200_vloss_0476.pt'

model = VAE(in_out_dim=513, hidden_dim=128, latent_dim=16 , num_hidden_layers=1).to(device)
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
model.eval()
for param in model.parameters():
    param.requires_grad = False


# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

processed_data_dir = os.path.join('data',dataset_size,'processed/')
input_speech_dir = os.path.join('data',dataset_size,'raw/')

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

model_data_dir = 'data/' + dataset_size + '/models/' + model_name + '/'


# Create file list
file_paths = speech_list(input_speech_dir=input_speech_dir,
                            dataset_type=dataset_type)

for i, file_path in tqdm(enumerate(file_paths)):
    
    # Read files
    s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav', dtype='float32') # clean speech
    x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav', dtype='float32') # mixture

    x_t = x_t/np.max(x_t)
    s_t = s_t/np.max(x_t)
    win = np.sin(np.arange(.5,1024-.5+1)/1024*np.pi) # sine analysis window
    Y = stft(x_t, n_fft=1024, hop_length=256, win_length=1024, window=win)
    X = torch.tensor(np.power(np.abs(Y), 2)).to(device)

    reconstruction, _, _, _ = model(X.T)   
    reconstruction = reconstruction.cpu().numpy()

    # Transpose to match librosa.display
    reconstruction = reconstruction.T

    # Transform to dB
    x_psd = X.cpu().numpy()
    x_psd = librosa.core.power_to_db(x_psd)          

    s_tf = stft(s_t, n_fft=1024, hop_length=256, win_length=1024, window=win)
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
    
    # Load input SNR
    all_snr_db = read_dataset(processed_data_dir, dataset_type, 'snr_db')
    all_snr_db = np.array(all_snr_db)
    
    # put all metrics in the title of the figure
    title = "Input SNR = {:.1f} dB \n" \
        "".format(all_snr_db[i])

    fig.suptitle(title, fontsize=40)

    # Save .wav files
    output_path = model_data_dir + file_path
    output_path = os.path.splitext(output_path)[0]

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    fig.savefig(output_path + '_recon.png')
