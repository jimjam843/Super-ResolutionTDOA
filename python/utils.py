from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import gmean
from typing import Any

# Types
Any1D = np.ndarray[tuple[int], np.dtype[Any]]
Float1D = np.ndarray[tuple[int], np.dtype[float]]
Complex1D = np.ndarray[tuple[int], np.dtype[np.complex128]]
Int1D = np.ndarray[tuple[int], np.dtype[int]]

Any2D = np.ndarray[tuple[int,int], np.dtype[Any]]
Complex2D = np.ndarray[tuple[int,int], np.dtype[np.complex128]]

"""
DSP
"""

class Pulse(Enum):
    RECTANGE = 1
    RAISED_COSINE = 2

def rectangle_pulse(span: int, pulse_length: int) -> Float1D:
    """
    Generate a single rectangular pulse used for pulse shaping

    Args:
        span: filter length in symbols
        pulse_length: samples per pulse

    Returns:
        Sampled rectangular pulse
    """

    num_taps = span*pulse_length
    pulse = np.zeros(num_taps, dtype=float)
    ix0 = num_taps // 2 - pulse_length // 2
    pulse[ix0:ix0+pulse_length] = 1.0

    return pulse

def raised_cosine_pulse(span: int, pulse_length: int, beta: float = 0.35) -> Float1D:
    """
    Generate a raised cosine pulse used for pulse shaping
    See: https://pysdr.org/content/pulse_shaping.html#raised-cosine-filter for equation

    Args:
        span: filter length in symbols
        pulse_length: smaples per pulse
        beta: filter roll-off factor

    Returns:
        Samples raised cosine pulse
    """

    num_taps = span*pulse_length
    t = np.arange(num_taps) - (num_taps-1)//2
    pulse = np.sinc(t / pulse_length) * np.cos(np.pi * beta * t / pulse_length) \
        / (1 - (2 * beta * t / pulse_length)**2)

    return pulse

def random_qpsk_data(num_symbols: int, samples_per_symbol: int, pulse_shape: Pulse) -> Complex1D:
    """
    Generate a QPSK signal of random symbols

    Args:
        num_symbols: number of symbols in output data
        samples_per_symbol: symbol length in samples

    Returns:
        complex samples of QPSK modulated random data
    """

    syms_r = np.random.choice([-1,1], size=num_symbols)
    syms_i = np.random.choice([-1,1], size=num_symbols)

    num_samples = num_symbols * samples_per_symbol
    stuffed_syms = np.zeros(num_samples, dtype=np.complex128)
    stuffed_syms[::samples_per_symbol] = syms_r + 1j * syms_i

    match pulse_shape:
        case Pulse.RECTANGE:
            pulse = rectangle_pulse(10, samples_per_symbol)

        case Pulse.RAISED_COSINE:
            pulse = raised_cosine_pulse(10, samples_per_symbol)

    return np.convolve(stuffed_syms, pulse, 'same')

def lfm_data(f1: float, f2: float, num_samples: int) -> Complex1D:
    """
    Generate a linear frequency modulated signal

    Args:
        f1: Starting frequency, in cycles/sample
        f2: Ending frequency, in cycles/sample
        num_samples: Length of signal in samples

    Returns:
        complex samples of a LFM signal
    """

    # chirp rate
    w = (f2-f1) / num_samples / 2
    t = np.arange(num_samples)
    return np.exp(1j * 2 * np.pi * (w * t ** 2 + f1 * t))

def power_spectral_density(x: Complex1D, fs: float = 1.0) -> Float1D:
    """
    Estimate the power spectral density of x using a normalized DFT

    Args:
        x: complex voltage samples
        fs: sample rate

    Returns:
        Estimated power spectral density (PSD)
    """

    N = x.shape[0]
    X = np.fft.fft(x)
    psd = (np.abs(X)**2) / (fs * N)

    fvec = np.linspace(-fs/2, fs/2, N)

    return fvec, np.fft.fftshift(psd)

def complex_gauss_noise(power: float, num_samples: int) -> Complex1D:
    """
    Generate a vector of zero-mean complex gaussian noise

    Args:
        power: noise power
        num_samples: number of samples to generate

    Return:
        Complex vector of noise samples
    """

    return np.sqrt(power) / np.sqrt(2) * np.random.randn(num_samples) + \
            1j * np.sqrt(power) / np.sqrt(2) * np.random.randn(num_samples)

def sensor(signal: Complex1D, num_samps: int, toas: list[int], snr_dbs: list[float]) -> Complex1D:
    """
    Simulate receive signal at one sensor

    Args:
        signal: the received signal
        num_samps: number of samples of sensor data (should be greater than length of signal vector)
        toas: list of time-of-arrival values for each signal component
        snr_dbs: list of SNRs for each signal component (len(snr_dbs) == len(toas))

    Returns:
        Simulated sampled signal at sensor
    """

    Pn = 1
    y0 = complex_gauss_noise(Pn, num_samps)
    for toa, snr_db in zip(toas, snr_dbs):
        Ps = Pn * 10 ** (snr_db/10)
        y0[toa:toa+len(signal)] += np.sqrt(Ps) * signal

    return y0

def tdoa_cc(y0: Complex1D, y1: Complex1D, lag_min: int, lag_max: int) -> tuple[Int1D, Complex1D]:
    """
    Estimate time-difference-of-arrival by computing cross-correlation between signal y0 and y1

    Args:
        y0: sampled signal at sensor 0
        y1: sampled signal at sensor 1
        lag_min: minimum lag index to compute response at
        lag_max: maximum lag index to compute response at

    Returns:
        Cross correlation response and corresponding vector of computed lags
    """

    N = min(len(y0), len(y1))

    r01 = np.convolve(y0, y1[::-1].conj(), 'full')
    r01 = np.abs(r01[N-1+lag_min:N-1+lag_max])
    r01 /= np.max(r01)
    lagvec = np.arange(lag_min, lag_max)

    return lagvec, r01

"""
MUSIC Paper
"""

def normalized_cross_spec_covariance_mat(y0: Complex1D, y1: Complex1D) -> Complex2D:
    """
    Computes the covariance matrix of the normalized cross spectrum between data vectors y0 and y1.

    The covariance matrix algorithm is described in Super-Resolution Time Delay Estimation in
    Multipath Environments Using Normalized Cross Spectrum by Zhong et. al. https://ieeexplore.ieee.org/document/6765235.

    The steps of the algorithm are as follows:
    1) Compute cross spectrum between y0 and y1 (S01)
    2) Compute DFT of autocorrelation of y0 (S00) and y1 (S11)
    3) Compute normalized cross spectrum (cross spectrum (1) divided by PSD of y0 and y1 (2))
    4) Compute covariance matrix of normalized cross spectrum
    """

    # Get normalized cross spectrum
    r01 = np.convolve(y0, y1[::-1].conj(), 'full')
    S01 = np.fft.fftshift(np.fft.fft(r01))
    r00 = np.convolve(y0, y0[::-1].conj(), 'full')
    S00 = np.fft.fftshift(np.fft.fft(r00))
    r11 = np.convolve(y1, y1[::-1].conj(), 'full')
    S11 = np.fft.fftshift(np.fft.fft(r11))
    h = S01/(np.sqrt(S00) * np.sqrt(S11))

    # Covariance matrix of normalized cross spectrum
    R = toeplitz(np.convolve(h, h[::-1].conj(), 'same'))

    return R

def tdoa_super(y0: Complex1D, y1: Complex1D, lag_min: int, lag_max: int, k: int, D: int) -> tuple[Float1D, Float1D]:
    """
    Computes TDOA estimate between y0 and y1 using super resolution algorithm described
    in Super-Resolution Time Delay Estimation in Multipath Environments Using Normalized
    Cross Spectrum by Zhong et. al. https://ieeexplore.ieee.org/document/6765235.

    The steps of the algorithm are as follows:
    1) Compute normalized covariance matrix using normalized_cross_spec_covariance_mat()
    2) Eigen-decomposition of covariance matrix and seperate into signal and noise sub-spaces
        - Note: This step assumes the number of signal components in the cross spectrum (k) is known.
    3) Plug the noise subspace using equation 16 in paper
        - My understanding of this equation is we are projecting the noise subspace onto the vector space
            spanned by the frequency domain phase shifts due to all possible time delays. These spaces should
            be orthogonal where signal is present, so the projection will go to zero for these time delays.
            The equation takes the inverse of the projection, so the local maxima in our output
            should correspond to TDOAs.

    Args:
        y0: sampled signal at sensor 0
        y1: sampled signal at sensor 1
        lag_min: minimum lag index to compute response at
        lag_max: maximum lag index to compute response at
        k: expected number of signal components in TDOA estimate
        D: oversample factor used when computing TDOA estimate

    Returns:
        Cross correlation response and corresponding vector of computed lags
    """

    R = normalized_cross_spec_covariance_mat(y0, y1)

    # Seperate noise subspace
    _, evec = np.linalg.eig(R)
    G = evec[:,k:]

    lagvec = np.arange(D*lag_min, D*lag_max) / D
    num_lags = len(lagvec)

    # Estimate TDOA response
    m = R.shape[0]
    A = np.exp(-1j * 2 * np.pi * np.arange(m)[np.newaxis,:] * lagvec[:,np.newaxis] / m)
    P = np.zeros(num_lags)
    for ix in range(num_lags):
        tmp = A[ix].conj().T @ G
        P[ix] = 1 / (np.abs(tmp)**2).sum() ** 2
    P /= np.max(P)

    return lagvec, P

"""
Model Order Estimation
"""

def overlapping_windows(data: Any1D, window_size: int, overlap: int) -> Any2D:
    """
    Generates overlapping sliding windows from a 1D numpy array.

    Args:
        data: The input 1D array.
        window_size: The size of the sliding window.
        overlap: The number of elements to overlap between consecutive windows.

    Returns:
        A 2D array where each row is a window.
    """
    step = window_size - overlap
    return np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size)[::step]

def AIC(evals: Float1D) -> int:
    """
    Model order estimation using Akaike Information Criterion (AIC)
    https://en.wikipedia.org/wiki/Akaike_information_criterion

    Args:
        evals: sorted eigevalues of the covariance matrix (descending, signal first).

    Returns:
        The estimated model order k (number of signal components).
    """

    N = len(evals)
    aic = np.zeros(N, dtype=float)
    for k in range(N):
        L = np.mean(evals[k:]) / gmean(evals[k:])
        aic[k] = N*(N-k)*np.log(L) + k*(2*N-k)
    return np.argmin(aic).astype(int)

def MDL(evals: Float1D) -> int:
    """
    Model order estimation using Minimum Description Length (MDL)
    https://en.wikipedia.org/wiki/Minimum_description_length

    Args:
        evals: sorted eigenvalues of the covariance matrix (descending, signal first).

    Returns:
        The estimated model order k (number of signal components).
    """
    N = len(evals)
    M = N

    mdl = np.zeros(N, dtype=float)
    for k in range(N):
        L = np.mean(evals[k:]) / gmean(evals[k:])
        mdl[k] = M*(N-k)*np.log(L) + 0.5*k*(2*N-k) * np.log(M)

    return np.argmin(mdl).astype(int)

"""
Plotting
"""

def plot_iq(x: Complex1D, size: int, offset: int=0) -> None:
    """
    Plot in-phase and quadrature components of complex voltage samples

    Args:
        x: complex voltage samples to be plotted
        size: number of samples of x to include in plot
        offset: first index to plot of x
    """

    plt.title("Voltage Samples")
    plt.plot(np.real(x[offset:offset+size]), label='I')
    plt.plot(np.imag(x[offset:offset+size]), label='Q')
    plt.grid()
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

def plot_psd(x: Complex1D) -> None:
    """
    Plot power spectral density of complex voltage samples

    Args:
        x: complex voltage samples to be plotted
        fs: sample frequency of vector x
    """

    fvec, psd = power_spectral_density(x)

    log_psd = 10*np.log10(psd)
    plt.plot(fvec, log_psd)

    plt.title("Power Spectral Density")
    plt.ylim([np.max(log_psd)-60, np.max(log_psd)+5])
    plt.grid()
    plt.xlabel("Frequency (cycle/second)")
    plt.ylabel("dB")