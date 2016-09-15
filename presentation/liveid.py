#!/usr/bin/env python

import numpy as np
from scipy import ndimage, sparse
import librosa
from librosa.filters import cq_to_chroma
from librosa.beat import beat_track
from librosa import frames_to_samples
from librosa.util import normalize

__author__ = "Zafar Rafii, Aneesh Vartakavi"
__date__ = "11/25/15"


def parameters():

    # Default parameters
    sample_rate = 22050             # Sample rate in Hz
    time_resolution = 10            # Number of time frames per second
    frequency_resolution = 2        # Number of frequency channels per semitone
    minimum_frequency = 130.81      # Minimum frequency in Hz (=C3) (included)
    maximum_frequency = 2093.00     # Maximum frequency in Hz (=C7) (excluded)
    cqt_kernel = kernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)    # CQT kernel
    neighborhood_size = (25, 31)    # Neighborhood size in frequency channels and time frames for the median filtering
    pitch_shifts = 6                # Pitch shifts in frequency channels (+-)
    block_width = 30                # Block width in time frames for the similarity filtering

    audio_parameters = {'sample_rate': sample_rate,
                        'time_resolution': time_resolution,
                        'frequency_resolution': frequency_resolution,
                        'minimum_frequency': minimum_frequency,
                        'maximum_frequency': maximum_frequency,
                        'cqt_kernel': cqt_kernel,
                        'neighborhood_size': neighborhood_size,
                        'pitch_shifts': pitch_shifts,
                        'block_width': block_width}

    return audio_parameters


def fingerprint(audio_signal, audio_parameters):

    # Parameter(s)
    sample_rate = audio_parameters['sample_rate']
    time_resolution = audio_parameters['time_resolution']
    cqt_kernel = audio_parameters['cqt_kernel']
    neighborhood_size = audio_parameters['neighborhood_size']

    # Transform the signal into a log-scaled spectrogram using the CQT
    audio_spectrogram = spectrogram(audio_signal, sample_rate, time_resolution,
                                    cqt_kernel)

    # Convert the spectrogram to uint8 (to speed up medfilt2) (in Matlab!)
    audio_spectrogram = np.uint8(np.around(255*audio_spectrogram/np.amax(audio_spectrogram)))
    # Segment the spectrogram into a binary image using an adaptive thresholding method based on a median filtering
    audio_fingerprint = (audio_spectrogram >
                         ndimage.median_filter(audio_spectrogram, neighborhood_size, mode='reflect')).astype(float)

    return audio_spectrogram, audio_fingerprint


def fingerprint_chromagram(audio_signal, audio_parameters):
    # Parameter(s)
    sample_rate = audio_parameters['sample_rate']
    time_resolution = audio_parameters['time_resolution']
    cqt_kernel = audio_parameters['cqt_kernel']
    neighborhood_size = audio_parameters['neighborhood_size']
    frequency_resolution = audio_parameters['frequency_resolution']
    bins_per_octave = 24*frequency_resolution
    minimum_frequency = audio_parameters['minimum_frequency']
    # Transform the signal into a log-scaled spectrogram using the CQT
    audio_spectrogram = spectrogram(audio_signal, sample_rate, time_resolution,
                                    cqt_kernel)

    # Convert the spectrogram to uint8 (to speed up medfilt2) (in Matlab!)
    audio_spectrogram = np.uint8(np.around(255*audio_spectrogram/np.amax(audio_spectrogram)))
    audio_fingerprint = (audio_spectrogram >
                         ndimage.median_filter(audio_spectrogram, neighborhood_size, mode='reflect')).astype(float)

    chroma_filter = cq_to_chroma(audio_fingerprint.shape[0], bins_per_octave = bins_per_octave, fmin = minimum_frequency)
    chromagram = chroma_filter.dot(audio_fingerprint)

    return chromagram

def fingerprint_beat(audio_signal, audio_parameters):
    # Parameter(s)
    sample_rate = audio_parameters['sample_rate']
    tempo, beats = beat_track(audio_signal, sample_rate)
    
    #time_resolution = audio_parameters['time_resolution']
    time_resolution = 3*np.floor(tempo / 60)
    if time_resolution == 0:
        time_resolution = 3
    cqt_kernel = audio_parameters['cqt_kernel']
    neighborhood_size = audio_parameters['neighborhood_size']

    # Transform the signal into a log-scaled spectrogram using the CQT
    audio_spectrogram = spectrogram(audio_signal, sample_rate, time_resolution,
                                    cqt_kernel)

    # Convert the spectrogram to uint8 (to speed up medfilt2) (in Matlab!)
    audio_spectrogram = np.uint8(np.around(255*audio_spectrogram/np.amax(audio_spectrogram)))

    # Segment the spectrogram into a binary image using an adaptive thresholding method based on a median filtering
    audio_fingerprint = (audio_spectrogram >
                         ndimage.median_filter(audio_spectrogram, neighborhood_size, mode='reflect')).astype(float)

    return audio_spectrogram

def fingerprint_similarity(song_fingerprint, cover_fingerprint, audio_parameters):
    # Parameter(s)
    time_resolution = audio_parameters['time_resolution']
    block_width = audio_parameters['block_width']
    pitch_shifts = audio_parameters['pitch_shifts']

    # Number of frequency channels and time frames in the query
    (number_frequencies, number_times) = cover_fingerprint.shape

    # Number of time frames for the query after similarity filtering
    number_times = number_times-block_width

    similarity_matrices = []

    for shift_index in range(-pitch_shifts, pitch_shifts+1):
        # Query shifted
        cover_fingerprint_shift \
            = cover_fingerprint[np.maximum(shift_index, 0)
                                :np.minimum(number_frequencies-1+shift_index, number_frequencies-1), :]

        # Reference shifted (the other way around)
        song_fingerprint_shift \
            = song_fingerprint[np.maximum(-shift_index, 0)
                                    :np.minimum(number_frequencies-1-shift_index, number_frequencies-1), :]

        # Weighted Hamming similarity between every pair of block of subfingerprints between the query and the reference
        similarity_matrix = similarity(cover_fingerprint_shift, song_fingerprint_shift, block_width)
        # Similarity matrix binarized
        similarity_matrix = similarity_matrix >= 0.5775
        similarity_matrix = similarity_matrix.astype(float)
        similarity_matrices.append(similarity_matrix)

    #stack = np.stack(similarity_matrices, axis=0)
    #max_sim = np.max(stack, axis = 0)
    return similarity_matrices

def fingerprint_chromagram(audio_signal, audio_parameters):

    # Parameter(s)
    sample_rate = audio_parameters['sample_rate']
    time_resolution = audio_parameters['time_resolution']
    cqt_kernel = audio_parameters['cqt_kernel']
    neighborhood_size = audio_parameters['neighborhood_size']
    frequency_resolution = audio_parameters['frequency_resolution']
    bins_per_octave = 12*frequency_resolution
    minimum_frequency = audio_parameters['minimum_frequency']
    # Transform the signal into a log-scaled spectrogram using the CQT
    audio_spectrogram = spectrogram(audio_signal, sample_rate, time_resolution,
                                    cqt_kernel)

    # Convert the spectrogram to uint8 (to speed up medfilt2) (in Matlab!)
    audio_spectrogram = np.uint8(np.around(255*audio_spectrogram/np.amax(audio_spectrogram)))
    audio_fingerprint = (audio_spectrogram >
                         ndimage.median_filter(audio_spectrogram, neighborhood_size, mode='reflect')).astype(float)

    chroma_filter = cq_to_chroma(audio_spectrogram.shape[0], bins_per_octave = bins_per_octave, fmin = minimum_frequency)
    chromagram = chroma_filter.dot(audio_fingerprint)

    # Segment the spectrogram into a binary image using an adaptive thresholding method based on a median filtering
    #audio_fingerprint = (chromagram >
                         #ndimage.median_filter(chromagram, (3, 3), mode='reflect')).astype(float)
    
    return chromagram

def transchromagram(chroma):
    transition_matrix = np.zeros((chroma.shape[0], chroma.shape[0]))
    for c in range(chroma.shape[1] - 1):
        ci = chroma[:, c]
        cj = chroma[:, c + 1]
        for i, x in enumerate(ci):
            for j, y in enumerate(cj):
                transition_matrix[i, j] += x*y > 0
    return transition_matrix

def match(query_fingerprint, reference_fingerprint, audio_parameters):

    # Parameter(s)
    time_resolution = audio_parameters['time_resolution']
    block_width = audio_parameters['block_width']
    pitch_shifts = audio_parameters['pitch_shifts']

    # Number of frequency channels and time frames in the query
    (number_frequencies, number_times) = query_fingerprint.shape

    # Number of time frames for the query after similarity filtering
    number_times = number_times-block_width

    # Score and offset initialized
    score_value = 0
    offset_index = 0

    # Loop over the pitch shifting
    for shift_index in range(-pitch_shifts, pitch_shifts+1):
        # Query shifted
        query_fingerprint_shift \
            = query_fingerprint[np.maximum(shift_index, 0)
                                :np.minimum(number_frequencies-1+shift_index, number_frequencies-1), :]

        # Reference shifted (the other way around)
        reference_fingerprint_shift \
            = reference_fingerprint[np.maximum(-shift_index, 0)
                                    :np.minimum(number_frequencies-1-shift_index, number_frequencies-1), :]

        # Weighted Hamming similarity between every pair of block of subfingerprints between the query and the reference
        similarity_matrix = similarity(query_fingerprint_shift, reference_fingerprint_shift, block_width)
        # Similarity matrix binarized
        similarity_matrix = similarity_matrix > 0.6
        similarity_matrix = similarity_matrix.astype(float)
        # Simple Dynamic Time Warping
        maximum_value, maximum_index = sdtw(similarity_matrix)

        # If maximum value higher than current score, update current score and offset
        if maximum_value > score_value:
            score_value = maximum_value
            offset_index = maximum_index+1

    # Convert offset to index in seconds
    offset_index = offset_index/float(time_resolution)

    return score_value, offset_index


def kernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency):

    # Number of frequency channels per octave ("float" for future computations)
    octave_resolution = float(12*frequency_resolution)

    # Constant ratio of frequency to resolution
    quality_factor = 1/(2**(1/octave_resolution)-1)

    # Number of frequency channels for the CQT
    number_frequencies = int(round(octave_resolution*np.log2(maximum_frequency/minimum_frequency)))

    # Window length for the FFT (next higher power of 2)
    fft_length = int(2**np.ceil(np.log2(quality_factor*sample_rate/minimum_frequency)))

    # Initialize the CQT kernel (better than concatenate)
    cqt_kernel = np.empty([number_frequencies, fft_length], dtype=complex)

    # Loop over the frequency channels
    for frequency_index in range(0, number_frequencies):

        # Frequency value (in Hz)
        frequency_value = minimum_frequency*2**(frequency_index/octave_resolution)

        # Window length (nearest odd value for symmetry)
        window_length = 2*round(quality_factor*sample_rate/frequency_value/2)+1

        # Temporal kernel (symmetric)
        temporal_kernel = np.hamming(window_length) * \
            np.exp(2*np.pi*1j*quality_factor *
                   np.arange(-(window_length-1)/2, (window_length-1)/2+1)/window_length)/window_length

        # Pre zero-padding to center FFT (FFT does post zero-padding) ("insert" faster than "append" or "pad")
        temporal_kernel = np.insert(temporal_kernel, 0, np.zeros(np.ceil((fft_length-window_length)/2)))

        # Spectral kernel (mostly real, discard imaginary part)
        spectral_kernel = np.fft.fft(temporal_kernel, fft_length)

        # Make the spectral kernels sparser
        spectral_kernel[np.abs(spectral_kernel) < 0.01] = 0

        # Add the spectral kernels
        cqt_kernel[frequency_index, :] = spectral_kernel

    # Make the CQT kernel sparse (faster than using sparse.lil_matrix at the initialization)
    cqt_kernel = sparse.csr_matrix(cqt_kernel)

    # From Parseval's theorem
    cqt_kernel = np.conj(cqt_kernel)/fft_length

    return cqt_kernel


def spectrogram(audio_signal, sample_rate, time_resolution, cqt_kernel):

    # Number of time samples per time frame
    step_length = np.floor(sample_rate/time_resolution)

    # Number of time frames
    number_times = int(np.floor(np.size(audio_signal)/step_length))

    # Number of frequency channels and FFT length
    (number_frequencies, fft_length) = cqt_kernel.shape

    # Zero-padding to center the CQT
    audio_signal = np.pad(audio_signal,
                          (int(np.ceil((fft_length-step_length)/2)), int(np.floor((fft_length-step_length)/2))),
                          'constant')

    # Initialize the CQT spectrogram
    audio_spectrogram = np.empty([number_frequencies, number_times])
    # Loop over the time frames
    for time_index in range(0, number_times):

        # CQT with kernel (magnitude)
        audio_spectrogram[:, time_index] \
            = abs(cqt_kernel * np.fft.fft(audio_signal[int(time_index*step_length):int(time_index*step_length)+fft_length]))

    return audio_spectrogram


def similarity(query_fingerprint, reference_fingerprint, block_width):

    # Number of frequency channels in both the query and the reference
    (number_frequencies, _) = query_fingerprint.shape

    # Hamming similarity for every pair of subfingerprints between the query and the reference
    # noinspection PyTypeChecker
    similarity_matrix = (np.dot(2*query_fingerprint.T-1, 2*reference_fingerprint-1)/float(number_frequencies)+1)/2

    # Hamming similarity for every pair of blocks of subfingerprints between the query and the reference
    # (ndimage.filters.correlate and cropping way faster than signal.correlate2d and 'valid'!)
    similarity_matrix = ndimage.filters.correlate(similarity_matrix, np.eye(block_width)/block_width)
    similarity_matrix = similarity_matrix[block_width/2:-block_width/2+1, block_width/2:-block_width/2+1]

    # Weights for every block of subfingerprints from the query
    query_weights = ndimage.filters.correlate(np.sum(query_fingerprint, axis=0, keepdims=True),
                                              np.ones((1, block_width)))
    query_weights = (2*query_weights[:, block_width/2:-block_width/2+1]/(number_frequencies*block_width)-1)**2+1

    # Weights for every block of subfingerprints from the reference
    reference_weights = ndimage.filters.correlate(np.sum(reference_fingerprint, axis=0, keepdims=True),
                                                  np.ones((1, block_width)))
    reference_weights = (2*reference_weights[:, block_width/2:-block_width/2+1]/(number_frequencies*block_width)-1)**2+1

    # Weighted Hamming similarity between every pair of blocks of subfingerprints between the query and the reference
    similarity_matrix = similarity_matrix/np.sqrt(query_weights.T*reference_weights)

    return similarity_matrix


def sdtw(similarity_matrix):

    # Number of time frames in the query
    number_times, _ = similarity_matrix.shape

    # Loop over the time frames of the query
    for time_index in range(number_times-1, 0, -1):

        # If the rows are not empty, simple DTW; else, copy the rows
        if any(similarity_matrix[time_index-1, :]):
            similarity_matrix[time_index-1, :] = similarity_matrix[time_index-1, :] + \
                ndimage.filters.maximum_filter(similarity_matrix[time_index, :], (5, 0), np.array([0, 0, 1, 1, 1]))
        else:
            similarity_matrix[time_index-1, 1:-2] = similarity_matrix[time_index, 2:-1]

    # Maximum index and value for the current pitch shifting
    maximum_index = np.argmax(similarity_matrix[0, :])

    # Maximum value normalized
    maximum_value = similarity_matrix[0, maximum_index]/float(number_times)

    return maximum_value, maximum_index
