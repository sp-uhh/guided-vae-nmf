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
from sklearn.metrics import f1_score

from python.processing.stft import stft
from python.processing.target import clean_speech_VAD, clean_speech_IBM
from python.models.models import Classifier
#from utils import count_parameters

from python.visualization import display_multiple_signals

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset

# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

# Labels
# labels = 'labels'
labels = 'vad_labels'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## IBM
quantile_fraction = 0.999
quantile_weight = 0.999

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Classifier
if labels == 'labels':
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_57.53'
    # x_dim = 513 # frequency bins (spectrogram)
    # y_dim = 513
    # h_dim_cl = [128, 128]
    # std_norm = True
    # eps = 1e-8

    # classif_name = 'classif_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_71.65'
    # x_dim = 513 # frequency bins (spectrogram)
    # y_dim = 513
    # h_dim_cl = [128, 128]
    # std_norm = False
    # eps = 1e-8

    classif_name = 'classif_batchnorm_before_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_59.58'
    x_dim = 513 # frequency bins (spectrogram)
    y_dim = 513
    h_dim_cl = [128, 128]
    batch_norm = True
    std_norm = False
    eps = 1e-8

if labels == 'vad_labels':
    classif_name = 'classif_VAD_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_0.21'
    x_dim = 513 # frequency bins (spectrogram)
    y_dim = 1
    h_dim_cl = [128, 128]
    std_norm = True
    batch_norm = False
    eps = 1e-8

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = 'data/' + dataset_size + '/models/' + classif_name + '/'
output_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

# Data normalization
if std_norm:
    # Load mean and variance
    mean = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_mean.npy')
    std = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_std.npy')

    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

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

    classifier = Classifier([x_dim, h_dim_cl, y_dim], batch_norm=batch_norm)
    classifier.load_state_dict(torch.load(classif_dir, map_location=cuda_device))
    if cuda: classifier = classifier.cuda()

    classifier.eval()
    for param in classifier.parameters():
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

        # Normalize power spectrogram
        if std_norm:
            x -= mean.T
            x /= (std + eps).T

        # Classify
        y_hat_soft = classifier(x)
        y_hat_hard = (y_hat_soft > 0.5).int()
        y_hat_hard = y_hat_hard.cpu().numpy()
        y_hat_hard = y_hat_hard.T  # Transpose to match librosa.display

        # plots of target / estimation
        s_tf = stft(s_t,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 dtype=dtype) # shape = (freq_bins, frames)

        if labels == 'labels':
            # binary mask
            target = clean_speech_IBM(s_tf,
                                    quantile_fraction=quantile_fraction,
                                    quantile_weight=quantile_weight)        

        if labels == 'vad_labels':
            # vad
            target = clean_speech_VAD(s_tf,
                                    quantile_fraction=quantile_fraction,
                                    quantile_weight=quantile_weight)

        # Transpose to match librosa.display
        x_tf = x_tf.T

        # F1-score
        f1score_s_hat = f1_score(target.flatten(), y_hat_hard.flatten(), average="binary")

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
            [s_t, s_tf, target], # clean speech
            [None, None, y_hat_hard]
            #[None, None, y_hat_soft]
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
        output_path = classif_data_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        fig.savefig(output_path + '_hard_mask.png')

if __name__ == '__main__':
    main()