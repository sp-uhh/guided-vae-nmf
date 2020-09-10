from python.data.EndtoEnd_VAD_data import speech_list
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
import soundfile as sf
from numpy.testing import assert_array_equal
from python.data.utils import open_file
import skvideo.io
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from python.visualization import display_waveplot, display_spectrogram
import numpy as np
import matplotlib.gridspec as grd


# Parameters
## Dataset
input_data_dir = 'data/complete/raw/'
output_data_dir = 'data/complete/processed/'
#fs = int(16e3) # Sampling rate
fs = int(8e3) # Sampling rate

## STFT
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.479  # hop size as a percentage of the window length
win = 'hann' # type of window

## Ideal binary mask
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
    file_paths = speech_list(input_data_dir=input_data_dir)

    for path in file_paths:
        #paths.append(path)
        x, fs_x = sf.read(input_data_dir + path, samplerate=None)
        x = x/np.max(np.abs(x))
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=fs,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent)
        


        # Check that TF rep. matches number of visual frames
        videodata = skvideo.io.vread(input_data_dir + os.path.splitext(path)[0] + '.avi') 
        
        # Raise warning
        if x_tf.shape[0] != videodata.shape[0]:
            warnings.warn('Audio and visual are not synchronized')

        spectrogram = np.power(np.abs(x_tf), 2)

        # Save spectrograms in npy
        output_path = output_data_dir + os.path.splitext(path)[0] + '_spectrum.npy'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        with open(output_path, 'wb') as f:
            np.save(f, spectrogram)

        # binary mask
        x_ibm = clean_speech_IBM(x_tf,
                                 quantile_fraction=quantile_fraction,
                                 quantile_weight=quantile_weight)
        
        # Save targets in npy
        output_path = output_data_dir + os.path.splitext(path)[0] + '_target.npy'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, 'wb') as f:
            np.save(f, x_ibm)

        # Plot waveplot + spectrogram + binary mask
        fig = plt.figure(figsize=(20,25))

        # create a 2 X 2 grid
        gs = grd.GridSpec(3, 2,
                        height_ratios=[5,10,10],
                        width_ratios=[10,0.5],
                        wspace=0.1,
                        hspace=0.3,
                        left=0.08)

        # line plot
        ax = plt.subplot(gs[0])
        display_waveplot(x, fs, xticks_sec, fontsize)

        # image plot
        ax = plt.subplot(gs[2])
        display_spectrogram(x_tf, True, vmin, vmax, fs, wlen_sec, hop_percent, xticks_sec, 'magma', fontsize)

        # color bar in it's own axis
        colorAx = plt.subplot(gs[3])
        cbar = plt.colorbar(cax=colorAx, format='%+2.0f dB')

        # image plot
        ax = plt.subplot(gs[4])
        display_spectrogram(x_ibm, False, 0, 1, fs, wlen_sec, hop_percent, xticks_sec, 'Greys_r', fontsize)

        # color bar in it's own axis
        colorAx = plt.subplot(gs[5])
        plt.colorbar(cax=colorAx, format='%0.1f')

        # Save figure
        output_path = output_data_dir + os.path.splitext(path)[0] + '_fig.png'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        fig.savefig(output_path)
    
    print("data is stored in " + output_data_dir)
    
    #Open output directory
    open_file(output_data_dir)
        
if __name__ == '__main__':
    main()