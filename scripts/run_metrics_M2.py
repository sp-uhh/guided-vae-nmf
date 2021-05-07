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
import tempfile

from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM, clean_speech_VAD

from python.metrics import energy_ratios, compute_stats, compute_stats_noisnr
from pystoi import stoi
from pesq import pesq
from uhh_sp.evaluation import polqa
# from sklearn.metrics import f1_score
from python.models.utils import f1_loss

from python.visualization import display_multiple_signals

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset

# Settings
dataset_type = 'test'

# dataset_size = 'subset'
dataset_size = 'complete'

# Labels
# labels = 'labels'
labels = 'vad_labels'

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'
eps = 1e-8

## Ideal binary mask
if labels == 'labels':
    quantile_fraction = 0.98
    quantile_weight = 0.999

if labels == 'vad_labels':
    ## Voice activitiy detection
    quantile_fraction = 0.999
    quantile_weight = 0.999

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Stats
confidence = 0.95 # confidence interval

if labels == 'labels':
    # M2
    # model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_417.69'
    model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_098_vloss_414.57'

    # classifier
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_57.53'
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_073_vloss_56.43'
    classif_name = 'oracle_classif'
    # classif_name = 'timo_classif'

if labels == 'vad_labels':
    # M2
    # model_name = 'M2_VAD_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_465.98'
    # model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_085_vloss_487.80'
    # model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_087_vloss_482.93'
    model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_087_vloss_482.93'

    # classifier
    # classif_name = 'classif_VAD_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_0.21'
    classif_name = 'classif_VAD_qf0.999_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_090_vloss_0.23'
    # classif_name = 'oracle_classif'
    # classif_name = 'ones_classif'
    # classif_name = 'zeros_classif'
    # classif_name = 'timo_vad_classif'


# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
model_data_dir = os.path.join('data', dataset_size, 'models_wsj0', model_name, classif_name + '/') # Directory where estimated data is stored


def compute_metrics_utt(args):
    # Separate args
    file_path, snr_db = args[0], args[1]

    # Read files
    s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
    n_t, fs_n = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_n.wav') # noise
    x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
    s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(file_path)[0] + '_s_est.wav') # est. speech

    # compute metrics
    #TODO: potential pb with SI-SIR --> compute segmental SI-SDR
    ## SI-SDR, SI-SAR, SI-SNR
    si_sdr, si_sir, si_sar = energy_ratios(s_hat=s_hat_t, s=s_t, n=n_t)

    ## STOI (or ESTOI?)
    stoi_s_hat = stoi(s_t, s_hat_t, fs, extended=True)

    ## PESQ
    pesq_s_hat = pesq(fs, s_t, s_hat_t, 'wb') # wb = wideband
    
    ## POLQA
    # polqa_s_hat = polqa(s, s_t, fs)
    # all_polqa.append(polqa_s_hat)

    ## F1 score
    # ideal binary mask
    y_hat_hard = torch.load(model_data_dir + os.path.splitext(file_path)[0] + '_ibm_hard_est.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)
    # y_hat_hard = torch.load(model_data_dir + os.path.splitext(file_path)[0] + '_ibm_soft_est.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)
    # y_hat_hard = (y_hat_hard > 0.5).T # Transpose to match target y, shape = (freq_bins, frames)
    y_hat_hard = y_hat_hard.T # Transpose to match target y, shape = (freq_bins, frames)

    # TF representation
    s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)

    if labels == 'labels':
        y = clean_speech_IBM(s_tf,
                                quantile_fraction=quantile_fraction,
                                quantile_weight=quantile_weight)
    if labels == 'vad_labels':
        y = clean_speech_VAD(s_tf,
                        quantile_fraction=quantile_fraction,
                        quantile_weight=quantile_weight)

    # Convert y to Tensor for f1-score
    y_hat_hard = y_hat_hard.int()
    y = torch.LongTensor(y)

    accuracy, precision, recall, f1score_s_hat = f1_loss(y.flatten(), y_hat_hard.flatten(), epsilon=1e-12)

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
        [s_t, s_tf, y.numpy()], # clean speech
        [s_hat_t, s_hat_tf, y_hat_hard.numpy()]
    ]
    fig = display_multiple_signals(signal_list,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize)
    
    # put all metrics in the title of the figure
    title = "Input SNR = {:.1f} dB \n" \
        "SI-SDR = {:.1f} dB,  " \
        "SI-SIR = {:.1f} dB,  " \
        "SI-SAR = {:.1f} dB\n" \
        "STOI = {:.2f},  " \
        "PESQ = {:.2f} \n" \
        "Accuracy = {:.3f},  "\
        "Precision = {:.3f},  "\
        "Recall = {:.3f},  "\
        "F1-score = {:.3f}\n".format(snr_db, si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat,\
            accuracy, precision, recall, f1score_s_hat)

    fig.suptitle(title, fontsize=40)

    # Save figure
    fig.savefig(model_data_dir + os.path.splitext(file_path)[0] + '_fig.png')

    # Clear figure
    plt.close()

    metrics = [si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat,\
        accuracy, precision, recall, f1score_s_hat]
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
    
    # all_metrics = []
    # for arg in args:
    #     metrics = compute_metrics_utt(arg)
    #     all_metrics.append(metrics)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # Transform generator to list
    all_metrics = list(all_metrics)
    metrics_keys = ['SI-SDR', 'SI-SIR', 'SI-SAR', 'STOI', 'PESQ',\
        'ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_metrics,
                  all_snr_db=all_snr_db,
                  model_data_dir=model_data_dir,
                  confidence=confidence)

def main_polqa():

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                                dataset_type=dataset_type)

    # Fuse both list
    v_reference_paths = [processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav'
                            for file_path in file_paths]

    v_processed_paths = [model_data_dir + os.path.splitext(file_path)[0] + '_s_est.wav'
                            for file_path in file_paths]

    #  POLQA on short audio files
    extended_v_reference_paths = []
    extended_v_processed_paths = []

    for i, (file_path, v_reference_path, v_processed_path) in enumerate(zip(file_paths, v_reference_paths, v_processed_paths)):
        # Read files
        s_t, fs_s = sf.read(v_reference_path) # clean speech
        s_hat_t, fs_s_hat = sf.read(v_processed_path) # est. speech

        # if smaller, then convert to numpy array and pad, and remove from list
        if len(s_t) < 3 * fs:
            s_t = np.pad(s_t, (0, 3 * fs - len(s_t)))
            s_hat_t = np.pad(s_hat_t, (0, 3 * fs - len(s_hat_t)))
            
            # Remove from path list
            v_reference_paths.remove(v_reference_path)
            v_processed_paths.remove(v_processed_path)

            # Save as new files
            extended_v_reference_path = processed_data_dir + os.path.splitext(file_path)[0] + '_s_3sec.wav'
            extended_v_processed_path = model_data_dir + os.path.splitext(file_path)[0] + '_s_est_3sec.wav'

            sf.write(extended_v_reference_path, s_t, fs)
            sf.write(extended_v_processed_path, s_hat_t, fs)

            # Append to extended path list
            extended_v_reference_paths.append(extended_v_reference_path)
            extended_v_processed_paths.append(extended_v_processed_path)

    # Remove 3rd and last indices from extended list
    del extended_v_reference_paths[3]
    del extended_v_reference_paths[-1]

    del extended_v_processed_paths[3]
    del extended_v_processed_paths[-1]

    t1 = time.perf_counter()
    
    # path_all_polqa = polqa(v_reference=v_reference_paths[:1], v_processed=v_processed_paths[:1])
    # extended_all_polqa = polqa(v_reference=extended_v_reference_paths[3:4], v_processed=extended_v_processed_paths[3:4])
    path_all_polqa = polqa(v_reference=v_reference_paths, v_processed=v_processed_paths)
    # extended_all_polqa = polqa(v_reference=extended_v_reference_paths, v_processed=extended_v_processed_paths)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # Transform generator to list
    path_all_polqa = list(path_all_polqa)
    # extended_all_polqa = list(extended_all_polqa)

    with open(model_data_dir + 'path_all_polqa.txt', 'w') as f:
        for item in path_all_polqa:
            f.write("%s\n" % item)

    # with open(model_data_dir + 'extended_all_polqa.txt', 'w') as f:
    #     for item in extended_all_polqa:
    #         f.write("%s\n" % item)

    # Merge lists
    # all_polqa = path_all_polqa + extended_all_polqa
    all_polqa = path_all_polqa
    # all_polqa = extended_all_polqa
    metrics_keys = ['POLQA']

    # Compute & save stats
    compute_stats_noisnr(metrics_keys=metrics_keys,
                  all_metrics=all_polqa,
                  model_data_dir=model_data_dir,
                  confidence=confidence)
                  
if __name__ == '__main__':
    main()
    # main_polqa()