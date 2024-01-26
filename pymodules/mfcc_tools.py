import numpy as np
from python_speech_features import mfcc as psf_mfcc

def mfcc(signal, samplerate=44100, winlen=0.025, winstep=0.01, numcep=16,
         nfilt=18, nfft=None, lowfreq=0, highfreq=10000, preemph=0, ceplifter=18, appendEnergy=True,
         winfunc=lambda x:np.ones((x, ))):
    """Returns the MFCC of signal.

    This is a wrapper around `python_speech_features.mfcc` with experimentally
    chosen best default parameters for kps data."""
    return psf_mfcc(signal=signal,
                    samplerate=samplerate,
                    winlen=winlen,
                    winstep=winstep,
                    numcep=numcep,
                    nfilt=nfilt,
                    nfft=nfft,
                    lowfreq=lowfreq,
                    highfreq=highfreq,
                    preemph=preemph,
                    ceplifter=ceplifter,
                    appendEnergy=appendEnergy,
                    winfunc=winfunc,
                    )
