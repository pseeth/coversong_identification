from scipy.ndimage import median_filter, convolve
import datetime
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from liveid import *
from librosa import frames_to_samples
from scipy.spatial.distance import *
from librosa.effects import time_stretch
import matplotlib
from scipy.misc import imresize
from skimage.transform import warp
from skimage.transform import AffineTransform
from skimage.measure import compare_ssim as ssim
from scipy.ndimage.filters import gaussian_filter

#time_stretch_factors = [.7, .75, .8, .85, .9, .95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
#time_stretch_factors = [1.0]
#alphas = np.arange(.5, 2.05, .05)
#time_stretch_factors = [.5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
time_stretch_factors = np.arange(.5, 2.05, .05).tolist()

def rescale_fft(fft, alpha):
    transform = AffineTransform(scale = (alpha, 1), translation = (.5*(fft.shape[1] - alpha*fft.shape[1]), 0))
    return (1/alpha)*warp(fft, transform)

def figsize(x, y):
    matplotlib.rcParams['figure.figsize'] = (x, y)

def get_time_stretches(audio_signal):
    versions = []
    for t in time_stretch_factors:
        if t == 1:
            versions.append(audio_signal)
        else:
            versions.append(time_stretch(audio_signal, t))
    return versions

def fingerprint_fft(raw_fingerprint, audio_parameters, fingerprint_parameters):
    window_seconds = fingerprint_parameters['window_seconds']
    hop_factor = fingerprint_parameters['hop_factor']
    sample_rate = audio_parameters['sample_rate']
    window_length = window_seconds*audio_parameters['time_resolution']
    ffts = []
    start = 0
    while start <= raw_fingerprint.shape[1] - window_length:
        fft = np.fft.fft2(raw_fingerprint[:, start:start+window_length])
        fft = gaussian_filter(np.abs(np.fft.fftshift(fft)), sigma=.375)
        fft = np.abs(np.fft.fftshift(fft))
        #fft[fft < np.percentile(fft, 99)] = 0
        ffts.append(fft.flatten())
        start += int(window_length * hop_factor)   
    return np.array(ffts).T   

def rms(patch):
    return np.sqrt(np.mean(np.square(patch)))


def fft_sequence_similarity(ffts_query, ffts_reference):
    similarities = cdist(ffts_query.T, ffts_reference.T, 'euclidean')
    energy = np.mean(similarities)
    similarities = similarities / np.max(similarities)
    return 1 - similarities, energy

def find_diagonals(S):
    diagonals = []
    diagonal_locations = []
    for i in range(-max(S.shape) + 1, max(S.shape)):
        diag = S.diagonal(i)
        diag_indices = get_diagonal_indices(S.shape, i)
        current_length = 0
        current_weight = 0
        for j in range(0, len(diag)):
            if diag[j] > 0 and j != len(diag):
                current_length += 1
                current_weight += diag[j]
            else:
                if current_length > 1:
                    if j == len(diag) - 1 and diag[j] > 0:
                        current_weight += diag[j]
                        current_length += 1
                        diagonal_locations.append((diag_indices[j - current_length], diag_indices[j]))
                    else:
                        diagonal_locations.append((diag_indices[j - current_length], diag_indices[j-1]))
                    diagonals.append(current_length*current_weight)   
                current_length = 0
                current_weight = 0
    zipped = zip(diagonals, diagonal_locations)
    zipped = sorted(zipped, reverse=True, key = lambda x: x[0])
    return [x[0] for x in zipped], [x[1] for x in zipped]

def filter_offsets(diagonals, offsets):
    o = 1
    while o < len(offsets):
        if conflicts(offsets, offsets[o]):
            del offsets[o]
            del diagonals[o]
        else:
            o += 1
    return diagonals, offsets

def conflicts(diags, d2):
    for d1 in diags:
        if d1 != d2:
            if d2[0][0] >= d1[0][0] - 1 and d2[0][0] <= d1[1][0] + 1:
                if d2[1][0] >= d1[0][0] - 1 and d2[1][0] <= d1[1][0] + 1:
                    return True
    return False

def get_diagonal_indices(shape, k):
    indices = []
    if k >= 0:
        zig = 0
        while zig+k <= shape[1]:
            indices.append((zig, zig+k))
            zig += 1
    else:
        zag = 0
        k = abs(k)
        while zag+k < shape[0]:
            indices.append((zag+k, zag))
            zag += 1
    return indices

def filter_similarity_matrix(S):
    filtered = np.copy(S)
    filtered = filtered  / np.max(filtered)
    kernel = np.array([[1, -1], [-1, 1]])
    filtered = convolve(filtered, kernel)
    filtered[filtered < 0] = 0
    #kernel = np.diagflat(np.ones((1, 4)))
    #filtered = median_filter(filtered, footprint=kernel)
    return filtered

def show_similarity_matrix(S):
    plt.matshow(S, cmap='Greys')
    plt.yticks(range(S.shape[0]), [str(datetime.timedelta(seconds=np.round(window_seconds*hop_factor*i, 0))) 
                                   for i in range(S.shape[0])])
    plt.xticks(range(S.shape[1]), [str(datetime.timedelta(seconds=np.round(window_seconds*hop_factor*i, 0))) 
                                   for i in range(S.shape[1])], rotation='vertical')
    plt.grid()
    plt.ylabel('Cover song')
    plt.xlabel('Reference song')
    figsize(10, 10)
    plt.show()
