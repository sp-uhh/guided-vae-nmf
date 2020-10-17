import numpy as np
import librosa
#from torchaudio.transforms import Spectrogram
#import librosa.display
import math


"""
TF representation
"""


#TODO: write the spectrogram transform in pytorch (torchaudio)


def stft(x,
         fs=16e3,
         wlen_sec=50e-3,
         win='hann',
         hop_percent=0.25,
         center=True,
         pad_mode='reflect',
         pad_at_end=True,
         dtype='complex64'):
    """
    Arguments
        x: input (as time series)
            # librosa 0.6.1: x must be float
        fs: frequence sampling (in Hz)
        framesz: framesize (in seconds)
        hop: 1 - overlap (in [0,1])

    return
        f, t, Sxx (default: complex64 i.e. float32 for amplitude and phase)
        WARNING: null frequency is included in the spectrogram bins
    """
    if wlen_sec * fs != int(wlen_sec * fs):
        raise ValueError("wlen_sample of STFT is not an integer.")
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hopsamp = int(hop_percent * nfft) # hop size in samples

    # # Convert x to float (librosa 0.6.1)
    # if not np.issubdtype(x.dtype, np.floating):
    #     x = x.astype('float64')

    # Sometimes stft / istft shortens the ouput due to window size
    # so you need to pad the end with hopsamp zeros
    if pad_at_end:
        utt_len = len(x) / fs
        if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
            x_ = np.pad(x, (0,hopsamp), mode='constant')
        else:
            x_ = x

    Sxx = librosa.core.stft(y=x_,
                            n_fft=nfft,
                            hop_length=hopsamp,
                            win_length=None,
                            window=win,
                            center=center,
                            pad_mode=pad_mode,
                            dtype=dtype)
    return Sxx


def istft(Sxx,
          fs=16000,
          wlen_sec=50e-3,
          win='hann',
          hop_percent=0.25,
          center=True,
          dtype='float32',
          max_len=None):
    """
    Inverse STFT

    Sxx: input (as spectrogram)
    fs: frequence sampling (in Hz)
    framesz: framesize (in seconds)
    hop: 1 - overlap (in [0,1])
    max_len: shorten output time signal
             in case output time signal is longer than input time signal

    return
    t, x (default: float64)
    """
    if wlen_sec * fs != int(wlen_sec * fs):
        raise ValueError("wlen_sample of iSTFT is not an integer.")
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hopsamp = int(hop_percent * nfft) # hop size in samples

    x = librosa.core.istft(stft_matrix=Sxx,
                           hop_length=hopsamp,
                           win_length=nfft,
                           window=win,
                           center=center,
                           dtype=dtype,
                           length=max_len)

    if max_len:
        x = x[:int(max_len*fs)]
    return x