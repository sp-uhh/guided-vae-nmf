import os
import numpy as np
import torch
import soundfile as sf
import librosa

from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset
from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM

from python.metrics import energy_ratios
from pystoi import stoi
#from pesq import pesq
#from uhh_sp.evaluation import polqa
from sklearn.metrics import f1_score


from python.visualization import display_multiple_signals

# Settings
input_speech_dir = 'data/subset/raw/'
processed_data_dir = 'data/subset/processed/'
dataset_type = 'test'


eps = np.finfo(float).eps # machine epsilon

# Parameters
## Silence removal
top_db = 40

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## Ideal binary mask
quantile_fraction = 0.98
quantile_weight = 0.999

## Deep Generative Model
model_name = 'dummy_M2_10_epoch_010_vloss_108.79.pt'
model_data_dir = 'data/subset/models/' + model_name + '/'

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30




def main():
    # Load input SNR
    all_snr_db = read_dataset(processed_data_dir, dataset_type, 'snr_db')

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

    for i, file_path in enumerate(file_paths):

        # Read files
        s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
        n_t, fs_n = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_n.wav') # noise
        x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
        s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(file_path)[0] + '_s_est.wav') # est. speech

        # compute metrics
        ## SI-SDR, SI-SAR, SI-SNR
        si_sdr, si_sir, si_sar = energy_ratios(s_hat=s_hat_t, s=s_t, n=n_t)
        all_si_sdr.append(si_sdr)
        all_si_sir.append(si_sir)
        all_si_sar.append(si_sar)

        ## STOI (or ESTOI?)
        stoi_s_hat = stoi(s_t, s_hat_t, fs, extended=True)
        all_stoi.append(stoi_s_hat)

        ## PESQ
        # pesq_s_hat = pesq(fs, s_t, s_hat_t, 'wb')
        # all_pesq.append(pesq_s_hat)
        
        ## POLQA
        # polqa_s_hat = polqa(s, s_t, fs)
        # all_polqa.append(polqa_s_hat)

        ## F1 score
        # ideal binary mask
        y_hat = torch.load(model_data_dir + os.path.splitext(file_path)[0] + '_ibm_est.pt') # shape = (frames, freq_bins)
        y_seg = torch.t(y_hat > 0.5).cpu().numpy() # Transpose to match target y, shape = (freq_bins, frames)

        # TF reprepsentation
        s_tf = stft(s_t,
                 fs=fs,
                 wlen_sec=wlen_sec,
                 win=win,
                 hop_percent=hop_percent,
                 dtype=dtype) # shape = (freq_bins, frames)

        y = clean_speech_IBM(s_tf,
                             quantile_fraction=quantile_fraction,
                             quantile_weight=quantile_weight)

        f1score_s_hat = f1_score(y.flatten(), y_seg.flatten(), average="binary")
        all_f1score.append(f1_score)

        # plots of target / estimation
        # TF reprepsentation
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
            [s_t, s_tf, y], # clean speech
            [s_hat_t, s_hat_tf, y_seg]
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
            "STOI = {:.1f}  \n" \
            "F1-score = {:.2f} \n".format(all_snr_db[i], si_sdr, si_sir, si_sar, stoi_s_hat, f1score_s_hat)

        fig.suptitle(title, fontsize=40)

        # Save figure
        fig.savefig(model_data_dir + os.path.splitext(file_path)[0] + '_fig.png')

    
    #TODO: save each metric in 1 separate csv (si_sdr, si_sar, etc. )
    #TODO: ecart_type directly and store in stats

if __name__ == '__main__':
    main()