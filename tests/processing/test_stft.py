from librosa.util import example, example_info, list_examples
import soundfile as sf
import numpy as np

from python.processing.stft import stft, istft

from numpy.testing import assert_array_almost_equal


def test_stft_istft():
    """
    Check stft + istft gives the same input/ouput signal
    """
    ## STFT parameters
    wlen_sec = 80e-3 # window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    win = 'hann' # type of window

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

    # STFT
    x_tf = stft(x,
                fs=fs_x,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent)
    
    # iSTFT
    x_hat = istft(x_tf,
                  fs=fs_x,
                  wlen_sec=wlen_sec,
                  win=win,
                  hop_percent=hop_percent,
                  max_len=x_len)

    # Asser 
    assert_array_almost_equal(x, x_hat)

def test_concat_spectrograms():
    """
    Check that 2 spectrograms can be concatenated along axis=1
    i.e. spectogram.shape = (frames, frequency bins)
    i.e. spectogram.shape != (frequency bins, frames)
    """
    ## STFT parameters
    wlen_sec = 80e-3 # window length in seconds
    hop_percent = 0.25  # hop size as a percentage of the window length
    win = 'hann' # type of window

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
    x1, fs_x1 = sf.read(audio_path)
    x1_len = len(x1)

    audio_path = example('choice')
    x2, fs_x2 = sf.read(audio_path)
    x2_len = len(x2)

    assert fs_x1 == fs_x2

    # STFT
    x1_tf = stft(x1,
                fs=fs_x1,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent)

    x2_tf = stft(x2,
            fs=fs_x2,
            wlen_sec=wlen_sec,
            win=win,
            hop_percent=hop_percent)
    
    # Concat along axis=1
    np.concatenate([x1_tf, x2_tf], axis=1)