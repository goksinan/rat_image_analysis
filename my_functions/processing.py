import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from scipy import stats
from itertools import compress


## Design Second-order IIR Filter
def design_fancy_filter(fs, lc, hc=None, order=4, ftype=None, flag=False):
    """
    Design a butterworth filter.
    At least 2 arguments needed: filter cutoff, sample rate
    Ex:
       y = design_fancy_filter(16000, 30)
       y = design_fancy_filter(fs=10000, lc=30, hc=300, order=4, ftype=bandpass, flag=True)
       return: sos coefficients
    """
    nyq = fs/2

    if ftype is None or 'low':
        sos = signal.butter(order, lc/nyq, 'lowpass', output = 'sos')

    if ftype is 'bandpass':
        wn = [lc/nyq, hc/nyq]
        sos = signal.butter(order, wn, 'bandpass', output = 'sos')

    if ftype is 'bandstop':
        wn = [lc/nyq, hc/nyq]
        sos = signal.butter(order, wn, 'bandstop', output = 'sos')

    if ftype is 'high':
        sos = signal.butter(order, lc/nyq, 'highpass', output = 'sos')

    if flag is not False:
        w, h = signal.sosfreqz(sos)
        plt.figure(1)
        plt.plot(float(fs * 0.5 / np.pi) * w, abs(h), label='order={}'.format(order));
        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)');
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

    return sos


## Apply fancy filter
def apply_fancy_filter(data, fs, lc, hc=None, order=4, ftype=None):
    """
    Design a butterworth filter.
    At least 3 arguments needed: data, filter cutoff, sample rate
    Ex:
       y = apply_fancy_filter(x, 30, 16000)
       y = apply_fancy_filter(x, fs=10000, lc=30, hc=300, order=4, ftype=bandpass, flag=True)
    """
    sos = design_fancy_filter(fs, lc, hc, order, ftype)
    return signal.sosfiltfilt(sos, data, axis=0)


## Design Simple Low Pass Filter
def design_butter_lowpass(fs, cutoff, order=5, flag=False):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    if flag is not False:
        # Plot the frequency response.
        w, h = signal.freqz(b, a, worN=8000)
        plt.figure()
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

    return b, a


## Apply low-pass filter
def apply_butter_lowpass(data, fs, cutoff, order=5):
    b, a = design_butter_lowpass(fs, cutoff, order)
    return signal.filtfilt(b, a, data, axis=0)


##
def remove_saturated_channels(data, threshold, fs):
    """
    REMOVES SATURATED CHANNELS ENTIRELY
    Regardless of the duration of the saturated signal, the entire signal
    is removed. 3-level saturation detection:
        1. Based on a user specified threshold
        2. Based on range
    """
    # Remove baseline wandering
    lc = 10 # cutoff
    nyq = fs/2
    sos = signal.butter(4, lc/nyq, 'highpass', output = 'sos')
    data = signal.sosfiltfilt(sos, data, axis=0)

    original_channels = [i for i in range(len(data[0]))]

    TH = 3  # Threshold for quartile elimination method. The smaller, the more consevative

    # Bad channel detection based on saturation
    incl = [not (column>threshold).any() for column in data.T]
    data = data[:,incl]
    remaining_channels = list(compress(original_channels,incl))

    # Bad channel detection based on abnormal range
    doneFlag = False

    while doneFlag is False:
        nfeat = np.max(data,axis=0) - np.min(data,axis=0)
        nfeat = nfeat.round()
        # sorted(nfeat, reverse=True)
        niqr = stats.iqr(nfeat)
        nfbig = nfeat > np.median(nfeat) + TH*niqr
        nfbig = ~nfbig
        nfsml = nfeat < np.median(nfeat) - TH*niqr
        nfsml = ~nfsml
        nfall = list(nfbig & nfsml)
        data = data[:,nfall]
        if not any(nfall):
            remaining_channels = list(compress(remaining_channels,nfall))
            continue
        else:
            doneFlag = True

    return remaining_channels


##
def remove_saturated_parts(neural, threshold, fs):
    """
    To remove saturated part plus x ms before & x ms after
    Returns a logical index vector
    To stitch the good parts back together
    """
    # Window length to remove
    rem = int(fs*0.01)

    # Remove baseline wandering
    lc = 0.1 # cutoff
    nyq = fs/2
    sos = signal.butter(4, lc/nyq, 'highpass', output = 'sos')
    neural = signal.sosfiltfilt(sos, neural, axis=0)

    for ind, column in enumerate(neural.T):
        satpnts = abs(column)>threshold
        if any(satpnts):
            pos_sat, = np.where(satpnts>0)
            for i in pos_sat:
                if i >= len(column)-rem:
                    column[i-rem:] = 0
                elif i <= rem:
                    column[:i+rem] = 0
                else:
                    column[i-rem:i+rem] = 0
            neural[:,ind] = column

    keep = [all(row) for row in neural]
    return keep


##
def find_envelope(data, fs, N=512, hc=8):
    """
    Finds the envelope of signal using a windowing technique
    :rtype: object
    :param data: signal whose envelope to be found
    :param fs: sampling rate for low-pass filter
    :param N: window length (default=512)
    :param hc: corner frequency of smoothing (low pass) filter (default:8)
    :return: the envelope
    """
    data = np.array(data)
    shifts = int(np.floor(len(data)/N))
    for i in range(shifts):
        s = i * N
        e = s + N
        data[s:e] = np.ones(np.shape(data[s:e])) * np.max(data[s:e], axis=0)
        if len(data) - e < N:
            data[e:] = np.ones(np.shape(data[e:])) * np.max(data[e:], axis=0)

    env = apply_fancy_filter(data, fs, hc)
    return env