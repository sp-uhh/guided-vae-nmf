from python.dataset.csr1_wjs0_dataset import speech_list, write_dataset, read_dataset
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
import numpy as np
import soundfile as sf
from numpy.testing import assert_array_equal
import pytest
import os
import h5py as h5

@pytest.mark.parametrize('dataset_type',
[
    ('train'),
    ('validation'),
]
)
def test_write_read_frames(dataset_type):
    # Parameters
    ## Dataset
    input_speech_dir = 'data/subset/raw/'
    output_speech_dir = 'data/subset/processed/'
    output_data_dir = 'data/subset/h5/'
    data_dir = 'CSR-1-WSJ-0'
    output_h5_dir = output_data_dir + data_dir + '.h5'

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                             dataset_type=dataset_type)

    # Open hdf5 file
    #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
    with h5.File(output_h5_dir, 'r', rdcc_nbytes=1024**2*400, rdcc_nslots=10e6) as f:

        dx = f['X_' + dataset_type]
        dy = f['Y_' + dataset_type]

        ## STFT
        fs = f.attrs['fs'] # Sampling rate
        wlen_sec = f.attrs['wlen_sec'] # window length in seconds
        hop_percent = f.attrs['hop_percent'] # hop size as a percentage of the window length
        win = f.attrs['win'] # type of window
        dtype = f.attrs['dtype']

        ## Ideal binary mask
        quantile_fraction = f.attrs['quantile_fraction']
        quantile_weight = f.attrs['quantile_weight']
        frame_begin = 0
        frame_end = 0

        for path in file_paths:

            x, fs_x = sf.read(input_speech_dir + path, samplerate=None)

            # Cut burst at begining of file
            #x[:int(0.1*fs)] = x[int(0.1*fs):int(0.2*fs)]
            x = x[int(0.1*fs):]

            # Normalize audio
            x = x/(np.max(np.abs(x)))
            
            if fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # TF reprepsentation
            x_tf = stft(x,
                        fs=fs,
                        wlen_sec=wlen_sec,
                        win=win,
                        hop_percent=hop_percent,
                        dtype=dtype)

            spectrogram = np.power(abs(x_tf), 2)

            # binary mask
            label = clean_speech_IBM(x_tf,
                                    quantile_fraction=quantile_fraction,
                                    quantile_weight=quantile_weight)


            # Read h5 spectrogram
            frame_end += spectrogram.shape[1]
            h5_spectrogram = dx[:,frame_begin:frame_end]
            h5_label = dy[:,frame_begin:frame_end]
    
            # Assert stored data is same as spectrograms
            assert_array_equal(spectrogram, h5_spectrogram)
            assert_array_equal(label, h5_label)

            # Next iteration
            frame_begin += spectrogram.shape[1]