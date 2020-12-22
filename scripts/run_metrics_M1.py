import sys
sys.path.append('.')

import os
import numpy as np
import torch
import soundfile as sf
import librosa
import json
import matplotlib.pyplot as plt
import concurrent.futures # for multiprocessing
import time

from python.processing.stft import stft, istft

from python.metrics import energy_ratios, compute_stats
from pystoi import stoi
from pesq import pesq
#from uhh_sp.evaluation import polqa

from python.visualization import display_multiple_signals

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset

# Settings
dataset_type = 'test'

dataset_size = 'subset'
# dataset_size = 'complete'

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'
#eps = np.finfo(float).eps # machine epsilon
eps = 1e-8

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Stats
confidence = 0.95 # confidence interval

# Model
#model_name = 'dummy_M2_10_epoch_010_vloss_108.79'
# model_name = 'dummy_M2_alpha_5.0_epoch_100_vloss_466.72'
# model_name = 'M1_hdim_128_128_zdim_032_end_epoch_200/M1_epoch_085_vloss_479.69'
model_name = 'M1_hdim_128_128_zdim_032_end_epoch_200/M1_epoch_124_vloss_475.95'

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
model_data_dir = os.path.join('data', dataset_size, 'models', model_name + '/') # Directory where estimated data is stored


def compute_metrics_utt(args):
    # Separate args
    file_path, snr_db = args[0], args[1]

    # print(file_path)
    # Read files
    s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
    n_t, fs_n = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_n.wav') # noise
    x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
    s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(file_path)[0] + '_s_est.wav') # est. speech

    # compute metrics
    #TODO: potential pb with SI-SIR --> compute segmental SI-SDR
    ## SI-SDR, SI-SAR, SI-SNR
    si_sdr, si_sir, si_sar = energy_ratios(s_hat=s_hat_t, s=s_t, n=n_t)
    # all_si_sdr.append(si_sdr)
    # all_si_sir.append(si_sir)
    # all_si_sar.append(si_sar)

    ## STOI (or ESTOI?)
    stoi_s_hat = stoi(s_t, s_hat_t, fs, extended=True)
    # all_stoi.append(stoi_s_hat)

    ## PESQ
    pesq_s_hat = pesq(fs, s_t, s_hat_t, 'wb') # wb = wideband
    # all_pesq.append(pesq_s_hat)
    
    ## POLQA
    # polqa_s_hat = polqa(s, s_t, fs)
    # all_polqa.append(polqa_s_hat)

    # TF representation
    s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)

    # plots of target / estimation
    # TF representation
    x_tf = stft(x_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)

    s_hat_tf = stft(s_hat_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)                 

    ## mixture signal (wav + spectro)
    ## target signal (wav + spectro + mask)
    ## estimated signal (wav + spectro + mask)
    signal_list = [
        [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
        [s_t, s_tf, None], # clean speech
        [s_hat_t, s_hat_tf, None]
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
        "".format(snr_db, si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat)

    fig.suptitle(title, fontsize=40)

    # Save figure
    fig.savefig(model_data_dir + os.path.splitext(file_path)[0] + '_fig.png')

    # Clear figure
    plt.close()

    metrics = [si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat]
    return metrics

def main():
    # Load input SNR
    all_snr_db = read_dataset(processed_data_dir, dataset_type, 'snr_db')
    all_snr_db = np.array(all_snr_db)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                                dataset_type=dataset_type)

    # Fuse both list
    args = [[file_path, snr_db] for file_path, snr_db in zip(file_paths, all_snr_db)]

    t1 = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        all_metrics = executor.map(compute_metrics_utt, args)
    
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # Transform generator to list
    all_metrics = list(all_metrics)
    metrics_keys = ['SI-SDR', 'SI-SIR', 'SI-SAR', 'STOI', 'PESQ']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_metrics,
                  all_snr_db=all_snr_db,
                  model_data_dir=model_data_dir,
                  confidence=confidence)

if __name__ == '__main__':
    main()