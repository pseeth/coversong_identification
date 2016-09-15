from liveid import *
from joblib import Parallel, delayed
import pickle
import tempfile
import numpy as np
import os
import shutil
import librosa
import sys
from scipy.ndimage.filters import convolve
from fft_sequence_fingerprint import *

from utilities import analyze_results

reference_directory = 'datasets/youtube_covers/references/'
query_directory = 'datasets/youtube_covers/queries/'
reference_fingerprint_directory = 'fingerprints/youtube_covers_refs/'
query_fingerprint_directory = 'fingerprints/youtube_covers_queries/'
reference_titles = sorted([x[:-4] for x in os.listdir(reference_directory) if '.wav' in x or '.mp3' in x])
query_titles = sorted([x[:-4] for x in os.listdir(query_directory) if '.wav' in x or '.mp3' in x])

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_directory(reference_fingerprint_directory)
create_directory(query_fingerprint_directory)

        
query_indices = range(len(query_titles))
reference_indices = range(len(reference_titles))
ap = parameters()

query_fingerprints = {}
reference_fingerprints = {}

reference_fingerprint_parameters = {
        'window_seconds': 20,
        'hop_factor': .2
    }

query_fingerprint_parameters = {
        'window_seconds': 20,
        'hop_factor': .2
    }

def load_file(file_name):
    print 'Loading %s' % file_name 
    f = open(file_name)
    load = pickle.load(f)
    f.close()
    return load

def save_file(file_name, content):
    print 'Saving %s' % file_name
    f = open(file_name, 'w')
    pickle.dump(content, f)
    f.close()

def fingerprint_wav(source_directory, target_directory, f, ap, stretch):
    print 'Fingerprinting: ' + f
    song, sr = librosa.load(os.path.join(source_directory, f + '.mp3'))
    ap['sample_rate'] = sr
    if stretch:
        for scale in time_stretch_factors:
            ap['sample_rate'] = scale*sr
            song_fingerprint = fingerprint(song, ap)
            song_fft_fingerprint = fingerprint_fft(song_fingerprint, ap, reference_fingerprint_parameters)
            save_file(target_directory + f + str(scale), song_fft_fingerprint)
    else:
        song_fingerprint = fingerprint(song, ap)
        song_fft_fingerprint = fingerprint_fft(song_fingerprint, ap, query_fingerprint_parameters)
        save_file(target_directory + f, song_fft_fingerprint)
    
def refingerprint():
    Parallel(n_jobs = 16)(delayed(fingerprint_wav)(reference_directory, reference_fingerprint_directory, f, ap, True) for f in reference_titles)
    Parallel(n_jobs = 16)(delayed(fingerprint_wav)(query_directory, query_fingerprint_directory, f, ap, False) for f in query_titles)

def measure_distance(query_fingerprint, references):
    min_distance = np.inf
    for reference_fingerprint in references:
        if len(reference_fingerprint) != 0 and len(query_fingerprint) != 0:
	    sim_matrix, energy = fft_sequence_similarity(query_fingerprint, reference_fingerprint)
	    filtered = filter_similarity_matrix(sim_matrix)
	    diagonals, offsets = find_diagonals(filtered)
	    print offsets
	    significant_diagonals = np.sum(diagonals[:3])
	    distance = energy/(significant_diagonals + .01)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def fill_element(t, distances, ap):
    i, j = t
    query_title = query_titles[i]
    reference_title = reference_titles[j]
    query_fingerprint = query_fingerprints[query_title]
    references = reference_fingerprints[reference_title]
    distances[i, j] = measure_distance(query_fingerprint, references)
    output = 'Comparing %d - %s to %d - %s: %f' % (i, query_titles[i], j, reference_titles[j], distances[i, j])
    print output

def load_fingerprints():
    print 'Loading query fingerprints'
    query_fingerprints = {}
    for t in [query_titles[i] for i in query_indices]:
        query_fingerprints[t] = load_file(query_fingerprint_directory + t)
    reference_fingerprints = {}
    print 'Loading reference fingerprints'
    for t in [reference_titles[i] for i in reference_indices]:
        reference_fingerprints[t] = [load_file(reference_fingerprint_directory + t + str(x)) for x in time_stretch_factors]
    return query_fingerprints, reference_fingerprints
 
def calculate_distances(filename):  
    print filename
    path = tempfile.mkdtemp()
    distances_path = os.path.join(path, 'distances_path.mmap')
    distances = np.memmap(distances_path, dtype=float, shape=((len(query_titles), len(reference_titles))), mode='w+')

    tuples = [(i, j) for i in query_indices for j in reference_indices]

    Parallel(n_jobs = 16)(delayed(fill_element)(t, distances, ap) for t in tuples)
    save_file('%s.pickle' % (filename), distances)
    
    print 'Queries to references'
    print analyze_results(distances)

    print 'References to queries'
    print analyze_results(distances.T)
       
    return distances

if __name__ == '__main__':
    filename = sys.argv[1]
    _refingerprint = 'refingerprint' in sys.argv
    if _refingerprint:
        refingerprint()
    query_fingerprints, reference_fingerprints = load_fingerprints()
    calculate_distances(filename)

