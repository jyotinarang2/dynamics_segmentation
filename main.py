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
from change_point_list import vocal_file_paths_and_change_points, get_change_points_for_all_songs
import laplacian_decomposition
import evaluation


# change to whatever audio you want to try here
index = 1
audio_path = vocal_file_paths_and_change_points[index]["vocals"]

def process(path, hop_size, downsampling):
    
    # p_name = str(pathlib.Path(*path.parts[-2:-1]))
    d = {}

    L, a1, a2 = laplacian_decomposition.get_representations(str(path), hop_length=hop_size, downsampling=downsampling)

    for l in [9, 12, 15, 18, 21]:
        d[str(l)] = {}

        d[str(l)]["L"] = np.abs(laplacian_decomposition.compute_novelty(L, l=l))
        d[str(l)]["a1"] = np.abs(laplacian_decomposition.compute_novelty(a1, l=l))
        d[str(l)]["a2"] = np.abs(laplacian_decomposition.compute_novelty(a2, l=l))

    return d



# get representations and compute novelty
hop_size = 2048
sr = 16000
downsampling = 16000.0//hop_size
paths = []
change_point_list_of_songs = get_change_points_for_all_songs()


for song in vocal_file_paths_and_change_points:
    paths.append(song["vocals"])
# print(change_point_list['all_of_me.wav'])

results = {}
for path in paths:
    d = process(path, hop_size, downsampling)
    results[os.path.basename(path)] = d

predictions = evaluation.predict_boundaries(results, paths, hop_size, downsampling, sr)
dev, hit = evaluation.evaluate_closeness_of_change_points_to_predictions(predictions, results, change_point_list_of_songs)
evaluation.compute_best_case_for_all_parameters(dev, hit, results, predictions)
evaluation.compute_best_f1_for_all_parameters(dev, hit, results, predictions)
evaluation.compute_best_precision_for_all_parameters(dev, hit, results, predictions)
evaluation.compute_best_recall_for_all_parameters(dev, hit, results, predictions)
evaluation.compute_best_case_precision_for_each_song(dev, hit, results, predictions)
evaluation.compute_best_case_recall_for_each_song(dev, hit, results, predictions)
evaluation.compute_best_case_f1_for_each_song(dev, hit, results, predictions)


