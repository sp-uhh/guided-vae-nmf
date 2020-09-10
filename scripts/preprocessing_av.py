from python.data.EndtoEnd_VAD_data import speech_list
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
from python.processing.EndtoEnd_VAD_data import get_audio_ground_truth, get_video_ground_truth, get_ground_truth_binary_mask
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
import math

# Parameters
## Dataset
input_data_dir = 'data/complete/raw/'
output_data_dir = 'data/complete/processed/'

## Ground truth audio/video
timeDepth = 3 # timeDepth of the TCN

Audio_frame_Length = 319
sampling_rate = 8e3
GlobalFrameRate = 25.2462 # to downsample the audio (nb of audio frames per second)
audio_duration = 122 # in seconds (?)
audio_len_in_frames = math.floor(audio_duration * sampling_rate / Audio_frame_Length)

video_duration_in_frames = 3059 # equal to audio_len_in_frames

## STFT
#should match GlobalFrameRate (wlen and hop_percent)
wlen_sec = 64e-3 # window length in seconds
hop_percent = math.floor((1/(wlen_sec*GlobalFrameRate))*1e4)/1e4  # hop size as a percentage of the window length
win = 'hann' # type of window

## Ideal binary mask
quantile_fraction = 0.98
quantile_weight = 0.999

def main():

    # Create file list
    file_paths = speech_list(input_data_dir=input_data_dir)

    for path in file_paths:
        # speaker_id
        speaker_id = os.path.basename(path)
        speaker_id = os.path.splitext(speaker_id)[0]

        # Read and trim audio at the end (audio_duration)
        x, fs_x = sf.read(input_data_dir + path,
                          samplerate=None)
        
        # Trim end of audio
        x = x[:int(sampling_rate*audio_duration)]
        
        # Normalize audio
        x = x/np.max(np.abs(x))
        if sampling_rate != fs_x:
            raise ValueError('Unexpected sampling rate')

        # TF reprepsentation
        x_tf = stft(x,
                    fs=sampling_rate,
                    wlen_sec=wlen_sec,
                    win=win,
                    hop_percent=hop_percent)

        # binary mask
        x_ibm = clean_speech_IBM(x_tf,
                                 quantile_fraction=quantile_fraction,
                                 quantile_weight=quantile_weight)

        
        # Save targets in npy
        output_path = output_data_dir + os.path.splitext(path)[0] + '.npy'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, 'wb') as f:
            np.save(f, x_ibm)
        
        # Check if get_ground_truth works
        npy_x_ibm = get_ground_truth_binary_mask(speaker_id)
        assert_array_equal(npy_x_ibm, x_ibm)
        




        # audio_labels = get_audio_ground_truth(speaker_id, GlobalFrameRate, audio_duration)

        # # Check that TF rep. matches number of visual frames
        # videodata = skvideo.io.vread(input_data_dir + os.path.splitext(path)[0] + '.avi')
        # videodata = videodata[:video_duration_in_frames]
        # video_labels = get_video_ground_truth(speaker_id, video_duration_in_frames)

        
if __name__ == '__main__':
    main()