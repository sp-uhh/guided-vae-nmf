import os
import sys
import pickle
import numpy as np
import torch
import time
import soundfile as sf
import math

sys.path.append('.')

from python.dataset.ntcd_timit_dataset import speech_list
from python.processing.stft import stft, istft
from python.processing.target import noise_robust_clean_speech_IBM, noise_robust_clean_speech_VAD
from python.utils import count_parameters
from python.models.mcem import MCEM_M2
from python.models.models import DeepGenerativeModel, Classifier, Classifier2Classes

##################################### SETTINGS #####################################################

# Dataset
dataset_type = 'test'

# dataset_size = 'subset'
dataset_size = 'complete'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:1"
device = torch.device(cuda_device if cuda else "cpu")

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

# Hyperparameters 
# M2
model_name = 'ntcd_M2_IBM_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_087_vloss_657.21'
x_dim = 513 
y_dim = 513
z_dim = 16
h_dim = [128, 128]
eps = 1e-8

model_dir = os.path.join('models', model_name + '.pt')

## Classifier
classif_name = 'ntcd_classif_IBM_normdataset_hdim_128_128_end_epoch_500/Classifier_epoch_014_vloss_87.76'
#classif_name = 'ntcd_classif_IBM_2classes_normdataset_hdim_128_128_end_epoch_500/Classifier_epoch_017_vloss_88.03'
h_dim_cl = [128, 128]
std_norm = True

## VAD
vad_name = 'visual_vad_classif'

classif_dir = os.path.join('models', classif_name + '.pt')

if std_norm:
    # Load mean and variance
    mean = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_mean.npy')
    std = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_std.npy')

    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

# NMF
nmf_rank = 10

### MCEM settings
niter = 100 # results reported in the paper were obtained with 500 iterations 
nsamples_E_step = 10
burnin_E_step = 30
nsamples_WF = 25
burnin_WF = 75
var_RW = 0.01

# Data directories
input_speech_dir = os.path.join('data', dataset_size,'raw/')
output_data_dir = os.path.join('data', dataset_size, 'models', model_name, classif_name + '_' + vad_name + '/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_data_dir = os.path.join('data', dataset_size, 'models', vad_name, dataset_type)

#####################################################################################################

def main():
    file = open('output.log','w') 

    print('Load models')
    classifier = Classifier([x_dim, h_dim_cl, y_dim])
    classifier.load_state_dict(torch.load(classif_dir, map_location=cuda_device))
    if cuda: classifier = classifier.to(device)

    model = DeepGenerativeModel([x_dim, y_dim, z_dim, h_dim], None)
    model.load_state_dict(torch.load(model_dir, map_location=cuda_device))
    if cuda: model = model.to(device)

    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir, dataset_type=dataset_type)
    print('- Number of test samples: {}'.format(len(file_paths)))

    print('Start evaluation')
    start = time.time()
    elapsed = []
    for i, file_path in enumerate(file_paths):
        start_file = time.time()
        print('- File {}/{}'.format(i+1,len(file_paths)), end='\r')

        x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
        s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # mixture

        T_orig = len(x_t)
        # TF reprepsentation
        # Input should be (frames, freq_bibs)
        x_tf = stft(x_t,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win, 
                    hop_percent=hop_percent,
                    center=center,
                    pad_mode=pad_mode,
                    pad_at_end=pad_at_end,
                    dtype=dtype) # shape = (freq_bins, frames)
                                     
        # Transpose to match PyTorch
        x_tf = x_tf.T # (frames, freq_bins)       
        
        # Power spectrogram (transpose)
        x = torch.tensor(np.power(np.abs(x_tf), 2)).to(device)
        
        # Normalize power spectrogram
        if std_norm:
            x_classif = x - mean.T
            x_classif /= (std + eps).T

            y_hat_soft = classifier(x_classif) 
        else:
            y_hat_soft = classifier(x)   
        y_hat_hard = (y_hat_soft > 0.5).float()

        # TF representation
        s_tf = stft(s_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win, 
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end,
            dtype=dtype) # shape = (freq_bins, frames)

        # Visual VAD
        # Read estimated labels
        label_path = os.path.join(classif_data_dir, file_paths[0].split('/')[-3], os.path.splitext(file_paths[0].split('/')[-1])[0] + '.npy')
        y_vad_visual = np.load(label_path)
        y_vad_visual = y_vad_visual[None]

        # Trim at end
        y_vad_visual = y_vad_visual[:,:x_tf.shape[0]]
        y_vad_visual = torch.from_numpy(y_vad_visual.T).to(device)

        
        # Compute y_hat_hard x y_vad_true
        y_hat_hard_new = y_hat_hard * y_vad_visual

        # Encode
        _, Z_init, _ = model.encoder(torch.cat([x, y_hat_hard_new], dim=1))

        # NMF parameters are initialized outside MCEM
        N, F = x_tf.shape
        W_init = np.maximum(np.random.rand(F,nmf_rank), eps)
        H_init = np.maximum(np.random.rand(nmf_rank, N), eps)
        g_init = torch.ones(N).to(device)

        mcem = MCEM_M2(X=x_tf, W=W_init, H=H_init, g=g_init, Z=Z_init, y=y_hat_hard,
                            vae=model, device=device, niter=niter,
                            nsamples_E_step=nsamples_E_step,
                            burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF, 
                            burnin_WF=burnin_WF, var_RW=var_RW)
        cost = mcem.run()

        S_hat = mcem.S_hat #+ np.finfo(np.float32).eps
        N_hat = mcem.N_hat #+ np.finfo(np.float32).eps

        # iSTFT
        s_hat = istft(S_hat,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent,
                    center=center,
                    max_len=T_orig)

        n_hat = istft(N_hat,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent,
                    center=center,
                    max_len=T_orig)

        # Save .wav files
        output_path = output_data_dir + file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        sf.write(output_path + '_s_est.wav', s_hat, fs)
        sf.write(output_path + '_n_est.wav', n_hat, fs)
        
        # Save binary mask
        torch.save(y_hat_soft, output_path + ' _ibm_soft_est.pt')
        torch.save(y_hat_hard_new, output_path + '_ibm_hard_est.pt')
        
        end_file = time.time()
        elapsed.append(end_file - start_file)
        etc = (len(file_paths)-i-1)*np.mean(elapsed)

        print("                   average time per file: {:4.1f} s      ETC: {:d} h, {:2d} min, {:2d} s"\
            "".format(np.mean(elapsed), int(etc/(60*60)), int((etc/60) % 60), int(etc % 60)), end='\r')

    end = time.time()
    print('- File {}/{}   '.format(len(file_paths), len(file_paths)))
    print('                     total time: {:6.1f} s'.format(end-start))
        
if __name__ == '__main__':
    main()