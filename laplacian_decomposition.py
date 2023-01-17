"""
Segmentation code derivative of:
     https://github.com/bmcfee/lsd_viz (MIT license)
with subsequent modifications from:
    https://github.com/chrispla/hierarchical_structure (MIT license)
and further modifications in this notebook.
"""
import os
import pathlib  # for finding files in directories
import sys  # for showing file loading progress bar
import librosa  # for audio processing
import librosa.display  # for plotting
import numpy as np  # for matrix and signal processing
import scipy  # for matrix processing
import cv2  # for image scaling
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # for parallel processing
from scipy.spatial.distance import pdist, squareform # for eigenvector set distances
from statistics import median, mean  # for calculating median boundary estimation deviation
import essentia
from essentia.standard import *

#Experiment with essentia loudness computation using bark bands
def compute_energy_bark_bands(audio_data, hop_size, block_size, sr):
    spectrum = Spectrum()
    barkBands = BarkBands(numberBands=20, sampleRate=sr)
    magnitude_spectrum = []
    fundamental_frequency = []
    bark_bands_energy = []
    w = Windowing(type = 'blackmanharris62')
    fft = FFT() # this gives us a complex FFT
    for frame in FrameGenerator(audio_data, frameSize=block_size, hopSize=hop_size, startFromZero=True, lastFrameToEndOfFile=True):      
        mag_spec = spectrum(w(frame))
        bark_bands =  barkBands(mag_spec)
        bark_bands_energy.append(bark_bands)
    bark_bands_energy = np.array(bark_bands_energy)
    return bark_bands_energy


def compute_laplacian(path, bins_per_octave, n_octaves, hop_length, downsampling):
    """Compute the Laplacian matrix from an audio file using
    its Constant-Q Transform.

    Args:
        path: filepath (str)
        bins_per_octave: number of bins per octave for CQT calculation
        n_octaves: number of octaves for CQT calculation

    Returns:
        L: Normalized graph Laplacian matrix (np.array)
        
    """

    # load audio
    y, sr = librosa.load(path, sr=16000, mono=True)

    # Compute Constant-Q Transform in dB
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y,
                                                   sr=sr,
                                                   bins_per_octave=bins_per_octave,
                                                   n_bins=n_octaves * bins_per_octave,
                                                   hop_length=hop_length)),
                                                   ref=np.max)
    bark_bands_energy = compute_energy_bark_bands(y, hop_length, 2*hop_length, sr)
    bark_bands_energy = np.transpose(bark_bands_energy)

    # we won't beat-synchronize, to keep reference to user annoations without
    # depending on accuracy of beat tracking. We'll instead downsample the hopped
    # CQT and MFCCs by a factor of 15,625, to comform with the standard +-0.5
    # second tolerance for estimation
    #Cdim = cv2.resize(C, (int(np.floor(C.shape[1]/downsampling)), C.shape[0]))
    #15.625 = 0.5*16000/512, its too long for this task? 
    Cdim = cv2.resize(bark_bands_energy, 
                      (int(np.floor(bark_bands_energy.shape[1]/downsampling)), 
                                          bark_bands_energy.shape[0]))
    print(Cdim.shape)

    # stack 4 consecutive frames
    Cstack = librosa.feature.stack_memory(Cdim, n_steps=4)

    # compute weighted recurrence matrix
    R = librosa.segment.recurrence_matrix(Cstack, width=5, mode='affinity')
    
    # enhance diagonals with a median filter
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    Rf = librosa.segment.path_enhance(Rf, 15)

    # compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
    print(mfcc.shape)
    Mdim = cv2.resize(mfcc, (int(np.floor(mfcc.shape[1]/downsampling)), mfcc.shape[0]))
    print(Mdim.shape)

    # build the MFCC sequence matrix
    path_distance = np.sum(np.diff(Mdim, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # get the balanced combination of the MFCC sequence matric and the CQT
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path

    # compute the normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    print("")

    return L

# @jit(forceobj=True)
def decompose_laplacian(L, k1, k2):
    """Decompose Laplacian and make sets of its first integer k (k1 or k2) eigenvectors.
    For each set, compute its Euclidean self distance over time.

    Args:
        L: Laplacian matrix (np.array)
        k1: first-eigenvectors number for first set (int)
        k2: first-eigenvectors number for secomd set (int)

    Returns:
        distances: self distance matrix of each set of first eigenvectors (np.array, shape=(kmax-kmin, 512, 512))
    """

    # eigendecomposition
    evals, evecs = scipy.linalg.eigh(L)

    # eigenvector filtering
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    # normalization
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    # initialize set
    distances = []

    for k in [k1, k2]:
        # create set using first k normalized eigenvectors
        Xs = evecs[:, :k] / Cnorm[:, k-1:k]

        # get eigenvector set distances
        distance = squareform(pdist(Xs, metric='euclidean'))
        distances.append(distance)

    return np.asarray(distances)

# @jit(forceobj=True)
def get_representations(path, hop_length=512.0, downsampling=15.625):
    """Simple end-to-end call for getting the three representations
    
    Args:
        path: audio path (str)
        k1: first-eigenvectors number for first set (int)
        k2: first-eigenvectors number for secomd set (int)
    
    Returns:
        tuple:
            Laplacian: Normalized graph Laplacian matrix (np.array)
            approximation k1: self-distance matrix for k1 set (np.array)
            approximation k2: self-distance matrix for k2 set (np.array)
    """
    L = compute_laplacian(path=path, 
                          bins_per_octave=12, 
                          n_octaves=3,
                          hop_length=hop_length, 
                          downsampling=downsampling)

    d = decompose_laplacian(L=L, k1=4, k2=9)

    return (L, d[0], d[1])

"""
Checkerboard kernel and novelty function code with minor modifications from:
    FMP Notebooks, C4/C4S4_NoveltySegmentation.ipynb
    which is an implementation of:
        Jonathan Foote: Automatic audio segmentation using a measure of audio 
        novelty. Proceedings of the IEEE International Conference on Multimedia 
        and Expo (ICME), New York, NY, USA, 2000, pp. 452–455.
"""

# @jit(nopython=True)
def compute_kernel_checkerboard_gaussian(l=20, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel.

    Args:
        l: Parameter specifying the kernel size M=2*l+1 (int)
        var: Variance parameter determing the tapering (epsilon) (float)
        normalize: Normalize kernel (bool)

    Returns:
        kernel: Kernel matrix of size M x M (np.ndarray)
    """

    taper = np.sqrt(1/2) / (l * var)
    axis = np.arange(-l, l+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

# @jit(nopython=True)
def compute_novelty(S, l=20, var=0.5, exclude=True):
    """Compute novelty function from SSM.

    Args:
        S: SSM (np.ndarray)
        l: Parameter specifying the kernel size M=2*l+1  (int)
        var: Variance parameter determing the tapering (epsilon) (float)
        exclude: Sets the first l and last l values of novelty function to zero (bool)

    Returns:
        nov (np.ndarray): Novelty function
    """

    kernel = compute_kernel_checkerboard_gaussian(l=l, var=var)
    N = S.shape[0]
    M = 2*l + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, l, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([l, N])
        left = np.max([0, N-l])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov