import sys
sys.path.append('.')

import os
import soundfile as sf
import numpy as np
import math

from python.dataset.ntcd_timit_dataset import speech_list
from python.processing.stft import stft
from python.processing.target import noise_robust_clean_speech_IBM, noise_robust_clean_speech_VAD, clean_speech_IBM

from python.visualization import display_waveplot, display_spectrogram, \
    display_wav_spectro_mask, display_multiple_signals
from python.utils import open_file

# Parameters
## Dataset
input_speech_dir = 'data/complete/raw/'
output_data_dir = 'data/subset/processed/'
dataset_type = 'validation'
fs = int(16e3) # Sampling rate

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

    file_paths = [i for i in file_paths if "08F/straightcam/sa1.wav" in i]
    # file_paths = [i for i in file_paths if "18M/straightcam/sa1.wav" in i]
    
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
                    hop_percent=hop_percent,
                    center=center,
                    pad_mode=pad_mode,
                    pad_at_end=pad_at_end,
                    dtype=dtype) # shape = (freq_bins, frames)
        
        # binary mask
        # x_ibm = clean_speech_IBM(x_tf,
        #                          quantile_fraction=ibm_quantile_fraction,
        #                          quantile_weight=quantile_weight)

        x_ibm = noise_robust_clean_speech_IBM(x_tf,
                                              vad_quantile_fraction_begin=vad_quantile_fraction_begin,
                                              vad_quantile_fraction_end=vad_quantile_fraction_end,
                                              ibm_quantile_fraction=ibm_quantile_fraction,
                                              quantile_weight=quantile_weight)
        
        # # compute only VAD
        # x_vad = noise_robust_clean_speech_VAD(x_tf,
        #                                       quantile_fraction_begin=vad_quantile_fraction_begin,
        #                                       quantile_fraction_end=vad_quantile_fraction_end,
        #                                       quantile_weight=quantile_weight)
        # x_vad = x_vad[0] # shape = (frames)

        # Plot waveplot + spectrogram + binary mask
        fig = display_wav_spectro_mask(x, x_tf, x_ibm,
                                 fs=fs, vmin=vmin, vmax=vmax,
                                 wlen_sec=wlen_sec, hop_percent=hop_percent,
                                 xticks_sec=xticks_sec, fontsize=fontsize)
        
        # # Plot waveplot + spectrogram + vad
        # fig = display_wav_spectro_mask(x, x_tf,x_vad,
        #                          fs=fs, vmin=vmin, vmax=vmax,
        #                          wlen_sec=wlen_sec, hop_percent=hop_percent,
        #                          xticks_sec=xticks_sec, fontsize=fontsize)

        # signal_list = [
        #     [x, x_tf, x_ibm], # mixture: (waveform, tf_signal, no mask)
        #     [x, x_tf, x_ibm], # clean speech
        #     [x, x_tf, x_ibm]
        # ]
        # fig = display_multiple_signals(signal_list,
        #                     fs=fs, vmin=vmin, vmax=vmax,
        #                     wlen_sec=wlen_sec, hop_percent=hop_percent,
        #                     xticks_sec=xticks_sec, fontsize=fontsize)

        title = "quantile_fraction_begin = {:.4f} \n" \
                "quantile_fraction_end = {:.4f} \n" \
                "quantile_weight = {:.4f} \n".format(vad_quantile_fraction_begin, vad_quantile_fraction_end, quantile_weight)
        fig.suptitle(title, fontsize=40)

        # Save figure
        output_path = output_data_dir + os.path.splitext(path)[0] + '_fig_ibm.png'
        #output_path = output_data_dir + os.path.splitext(path)[0] + '_q{:.2f}_'.format(quantile_fraction) + '_fig_vad.png'
        #output_path = output_data_dir + os.path.splitext(path)[0] + '_fig_vad.png'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        fig.savefig(output_path)
    
    print("data is stored in " + output_data_dir)
    
    #Open output directory
    #open_file(output_data_dir)

if __name__ == '__main__':
    # for quantile_fraction in [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    #     main(quantile_fraction)
    main()