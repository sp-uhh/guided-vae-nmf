from python.dataset.csr1_wjs0_dataset import speech_list
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
import soundfile as sf
import numpy as np

from python.visualization import display_waveplot, display_spectrogram, \
    display_wav_spectro_mask, display_multiple_signals
import numpy as np
import os
from python.utils import open_file

# Parameters
## Dataset
input_speech_dir = 'data/subset/raw/'
output_data_dir = 'data/subset/processed/'
dataset_type = 'test'
fs = int(16e3) # Sampling rate

## STFT
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window

## IBM
quantile_fraction = 0.98
quantile_weight = 0.999

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

#eps = np.finfo(float).eps # machine epsilon

def main():
    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)
    
    for path in file_paths:
        x, fs_x = sf.read(input_speech_dir + path, samplerate=None)
        x = x/np.max(np.abs(x))
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent) # shape = (freq_bins, frames)

        # binary mask
        x_ibm = clean_speech_IBM(x_tf,
                                quantile_fraction=quantile_fraction,
                                quantile_weight=quantile_weight)

        # Plot waveplot + spectrogram + binary mask
        fig = display_wav_spectro_mask(x, x_tf, x_ibm,
                                 fs=fs, vmin=vmin, vmax=vmax,
                                 wlen_sec=wlen_sec, hop_percent=hop_percent,
                                 xticks_sec=xticks_sec, fontsize=fontsize)

        # signal_list = [
        #     [x, x_tf, x_ibm], # mixture: (waveform, tf_signal, no mask)
        #     [x, x_tf, x_ibm], # clean speech
        #     [x, x_tf, x_ibm]
        # ]
        # fig = display_multiple_signals(signal_list,
        #                     fs=fs, vmin=vmin, vmax=vmax,
        #                     wlen_sec=wlen_sec, hop_percent=hop_percent,
        #                     xticks_sec=xticks_sec, fontsize=fontsize)

        title = "Input SNR = {:.1f} dB \n" \
                   "SI-SDR = {:.1f} dB, " \
                   "SI-SIR = {:.1f} dB, " \
                   "SI-SAR = {:.1f} dB \n" \
                   "STOI = {:.1f}  \n" \
                   "F1-score = {:.2f} dB \n".format(-5, -5, -5, -5, -5, 0.99)
        fig.suptitle(title, fontsize=40)

        # Save figure
        output_path = output_data_dir + os.path.splitext(path)[0] + '_fig.png'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        fig.savefig(output_path)
    
    print("data is stored in " + output_data_dir)
    
    #Open output directory
    #open_file(output_data_dir)

if __name__ == '__main__':
    main()