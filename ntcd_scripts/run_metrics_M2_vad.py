import sys
sys.path.append('.')

import os
import numpy as np
import torch
import soundfile as sf
import librosa
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from python.dataset.ntcd_timit_dataset import speech_list, read_dataset
from python.processing.stft import stft, istft
from python.processing.target import noise_robust_clean_speech_VAD

from python.metrics import energy_ratios, mean_confidence_interval
from pystoi import stoi
from pesq import pesq
#from uhh_sp.evaluation import polqa
from sklearn.metrics import f1_score

from python.visualization import display_multiple_signals

# Settings
dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

#eps = np.finfo(float).eps # machine epsilon
eps = 1e-8

# Parameters
# ## Silence removal
# top_db = 40

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
## Ideal binary mask 
quantile_fraction = 0.98
quantile_weight = 0.999

## Noise robust VAD
vad_quantile_fraction_begin = 0.93
vad_quantile_fraction_end = 0.99
quantile_weight = 0.999

## Noise robust IBM
ibm_quantile_fraction = 0.999
quantile_weight = 0.999

## Hyperparameters
# M2
model_name = 'ntcd_M2_VAD_hdim_128_128_zdim_016_end_epoch_500/M2_epoch_126_vloss_690.53'

# classifier
# classif_name = 'ntcd_classif_VAD_normdataset_hdim_128_128_end_epoch_500/Classifier_epoch_006_vloss_0.55'
# classif_name = 'oracle_classif'
classif_name = 'visual_vad_classif'
# classif_name = 'ones_classif'
# classif_name = 'zeros_classif'
# classif_name = 'timo_vad_classif'

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Stats
confidence = 0.95 # confidence interval

# Data directories
model_data_dir = os.path.join('data', dataset_size, 'models', model_name, classif_name + '/')
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')

def main():
    # Load input SNR
    all_snr_db = read_dataset(processed_data_dir, dataset_type, 'snr_db')
    all_snr_db = np.array(all_snr_db)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    # 1 list per metric
    all_si_sdr = []
    all_si_sir = []
    all_si_sar = []
    all_stoi = []
    all_pesq = []
    all_polqa = []
    all_f1score = []

    for i, file_path in tqdm(enumerate(file_paths)):

        # Read files
        s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
        n_t, fs_n = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_n.wav') # noise
        x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
        s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(file_path)[0] + '_s_est.wav') # est. speech

        # compute metrics
        #TODO: potential pb with SI-SIR --> compute segmental SI-SDR
        ## SI-SDR, SI-SAR, SI-SNR
        si_sdr, si_sir, si_sar = energy_ratios(s_hat=s_hat_t, s=s_t, n=n_t)
        all_si_sdr.append(si_sdr)
        all_si_sir.append(si_sir)
        all_si_sar.append(si_sar)

        ## STOI (or ESTOI?)
        stoi_s_hat = stoi(s_t, s_hat_t, fs, extended=True)
        all_stoi.append(stoi_s_hat)

        ## PESQ
        pesq_s_hat = pesq(fs, s_t, s_hat_t, 'wb') # wb = wideband
        all_pesq.append(pesq_s_hat)
        
        ## POLQA
        # polqa_s_hat = polqa(s, s_t, fs)
        # all_polqa.append(polqa_s_hat)

        ## F1 score
        # ideal binary mask
        y_hat_hard = torch.load(model_data_dir + os.path.splitext(file_path)[0] + '_ibm_hard_est.pt') # shape = (frames, freq_bins)
        y_hat_hard = torch.t(y_hat_hard > 0.5).cpu().numpy() # Transpose to match target y, shape = (freq_bins, frames)
        y_hat_hard = y_hat_hard[0] # shape = (frames)

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

        y = noise_robust_clean_speech_VAD(s_tf,
                                            quantile_fraction_begin=vad_quantile_fraction_begin,
                                            quantile_fraction_end=vad_quantile_fraction_end,
                                            quantile_weight=quantile_weight)
        y = y[0] # shape = (frames)

        f1score_s_hat = f1_score(y, y_hat_hard, average="binary")
        all_f1score.append(f1score_s_hat)

        # plots of target / estimation
        # TF representation
        x_tf = stft(x_t,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win, 
                    hop_percent=hop_percent,
                    center=center,
                    pad_mode=pad_mode,
                    pad_at_end=pad_at_end,
                    dtype=dtype) # shape = (freq_bins, frames)

        s_hat_tf = stft(s_hat_t,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win, 
                    hop_percent=hop_percent,
                    center=center,
                    pad_mode=pad_mode,
                    pad_at_end=pad_at_end,
                    dtype=dtype) # shape = (freq_bins, frames)                 

        ## mixture signal (wav + spectro)
        ## target signal (wav + spectro + mask)
        ## estimated signal (wav + spectro + mask)
        signal_list = [
            [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
            [s_t, s_tf, y], # clean speech
            [s_hat_t, s_hat_tf, y_hat_hard]
        ]
        fig = display_multiple_signals(signal_list,
                            fs=fs, vmin=vmin, vmax=vmax,
                            wlen_sec=wlen_sec, hop_percent=hop_percent,
                            xticks_sec=xticks_sec, fontsize=fontsize)
        
        # put all metrics in the title of the figure
        title = "Input SNR = {:.1f} dB \n" \
            "SI-SDR = {:.1f} dB, " \
            "SI-SIR = {:.1f} dB, " \
            "SI-SAR = {:.1f} dB \n" \
            "STOI = {:.2f}, " \
            "PESQ = {:.2f} \n" \
            "F1-score = {:.3f} \n".format(all_snr_db[i], si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat, f1score_s_hat)

        fig.suptitle(title, fontsize=40)

        # Save figure
        fig.savefig(model_data_dir + os.path.splitext(file_path)[0] + '_fig.png')

        # Clear figure
        plt.close()

    # Confidence interval
    metrics = {
        'SI-SDR': all_si_sdr,
        'SI-SIR': all_si_sir,
        'SI-SAR': all_si_sar,
        'STOI': all_stoi,
        'PESQ': all_pesq,
        'F1-score': all_f1score
    }

    stats = {}
    
    # Print the names of the columns. 
    print ("{:<10} {:<10} {:<10}".format('METRIC', 'AVERAGE', 'CONF. INT.')) 
    for key, metric in metrics.items():
        m, h = mean_confidence_interval(metric, confidence=confidence)
        stats[key] = {'avg': m, '+/-': h}
        print ("{:<10} {:<10} {:<10}".format(key, m, h))
    print('\n')

    # Save stats (si_sdr, si_sar, etc. )
    with open(model_data_dir + 'stats.json', 'w') as f:
        json.dump(stats, f)

    # Metrics by input SNR
    for snr_db in np.unique(all_snr_db):
        stats = {}

        print('Input SNR = {:.2f}'.format(snr_db))
        # Print the names of the columns. 
        print ("{:<10} {:<10} {:<10}".format('METRIC', 'AVERAGE', 'CONF. INT.')) 
        for key, metric in metrics.items():
            subset_metric = np.array(metric)[np.where(all_snr_db == snr_db)]
            m, h = mean_confidence_interval(subset_metric, confidence=confidence)
            stats[key] = {'avg': m, '+/-': h}
            print ("{:<10} {:<10} {:<10}".format(key, m, h))
        print('\n')

        # Save stats (si_sdr, si_sar, etc. )
        with open(model_data_dir + 'stats_{:g}.json'.format(snr_db), 'w') as f:
            json.dump(stats, f)

if __name__ == '__main__':
    main()