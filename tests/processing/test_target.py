from librosa.util import example, example_info, list_examples
from python.processing.stft import stft
from python.processing.target import clean_speech_IBM
import numpy as np
import soundfile as sf

def test_clean_speech_IBM():
    """
    check that output of 'complex64' spectrogram is 'float32' mask
    """
    ## STFT parameters
    wlen_sec = 80e-3 # window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    win = 'hann' # type of window
    dtype = 'complex64'

    # Librosa examples
    # # AVAILABLE EXAMPLES
    # # --------------------------------------------------------------------
    # # brahms    	Brahms - Hungarian Dance #5
    # # choice    	Admiral Bob - Choice (drum+bass)
    # # fishin    	Karissa Hobbs - Let's Go Fishin'
    # # nutcracker	Tchaikovsky - Dance of the Sugar Plum Fairy
    # # trumpet   	Mihai Sorohan - Trumpet loop
    # # vibeace   	Kevin MacLeod - Vibe Ace

    # Take example signal from Librosa
    audio_path = example('brahms')
    x, fs_x = sf.read(audio_path)
    x_len = len(x)

    ## Ideal binary mask
    quantile_fraction = 0.98
    quantile_weight = 0.999

    # STFT
    x_tf = stft(x,
                fs=fs_x,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype)
    
    # binary mask
    x_ibm = clean_speech_IBM(x_tf,
                                quantile_fraction=quantile_fraction,
                                quantile_weight=quantile_weight)
    
    assert x_ibm.dtype == 'float32'
    assert np.unique(x_ibm).tolist() == [0., 1.]

#TODO: take masks from Heymann GitHub