import numpy as np
from scipy.fft import fft
from scipy.fft import dct

def get_features(recording, window_len, maxi, mini, feature):
    windows = create_windows(recording, window_len)
    features = np.array([compute_features(window, maxi, mini, feature) for window in windows])

    return features

def create_windows(recording, window_len):
    """
    take in a recording, and turn it into sliding window samples, with an increment of 10 ms (window per sample)
    """
    window_size = int(window_len / 10)
    return np.lib.stride_tricks.sliding_window_view(recording, window_size)

def compute_features(window, maxi, mini, feature):
    """
    compute all features for a single window, 18 total

    we use all time domain features, given the processing already applied on the dataset

    add a single movement label to features
    """

    mav = np.average(abs(window))           # mean absolute value, 1 scalar
    var = np.var(window)                    # variance, 1 scalar
    wl = compute_wl(window)                 # waveform length, 1 scalar
    hist = compute_hist(window, maxi, mini) # histogram of amplitude, 10 bins (10 values), log scale
    cc = compute_cc(window)                 # cepstral coefficients, 5 scalar

    feature_map = {
        "MAV": np.array([mav]),
        "VAR": np.array([var]),
        "WL": np.array([wl]),
        "HIST": hist,
        "CC": cc
    }
    if feature == "all":
        features = np.concatenate((np.array([mav, var, wl]), hist, cc))
    else:
        features = feature_map[feature]

    return features

def compute_wl(window):
    """
    waveform length: absolute sum changes in value
    """
    window = np.insert(window, 0, 0)
    val_change = np.array([window[i] - window[i-1] for i in range(len(window))])

    return np.sum(abs(val_change))

def compute_hist(window, max, min, num_bins = 10):
    """
    histogram of the window
    """
    bins = np.logspace(np.log10(min + 1e-10), np.log10(max), num_bins + 1)

    return np.histogram(window, bins=bins)[0]

def compute_cc(window, num_cc = 5):
    """
    cepstral coefficients
    """
    n = len(window)
    power_spectrum = np.abs(fft(window, n=n)) ** 2
    power_spectrum = power_spectrum[:n//2] # floor division, only get positive values

    log_power = np.log(power_spectrum + 1e-10) # add small value to avoid log(0)
    coeffs = dct(log_power, type=2, norm="ortho")

    return coeffs[:num_cc]