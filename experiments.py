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
from change_point_list import vocal_file_paths_and_change_points
import laplacian_decomposition

index = 12
audio_path = vocal_file_paths_and_change_points[index]["vocals"]
# store all variables and features in this experiment in a dictionary for easier retrieval
# c1: Laplacian, c2: approximation 1, c3: approximation 2
exp = {'c1': {}, 'c2': {}, 'c3': {}}

# get representations and compute novelty
hop_size = 2048
downsampling = 16000.0//hop_size
exp['c1']['rep'], exp['c2']['rep'], exp['c3']['rep'] = laplacian_decomposition.get_representations(audio_path, hop_size, downsampling)
for case in ['c1', 'c2', 'c3']:
    exp[case]['nov'] = np.abs(laplacian_decomposition.compute_novelty(exp[case]['rep'], l=21))

# we want to define a threshold value for choosing novelty peaks to consider as boundaries
# for this experiment, let's define it as 70% of the difference between the global maximum
# and the non-zero global minimum of novelties.
ptp = 0.2 
sr = 16000

for case in ['c1', 'c2', 'c3']:

    gmax = np.amax(exp[case]['nov'])
    gmin = np.amin(exp[case]['nov'][np.nonzero(exp[case]['nov'])])

    pt = gmin + (ptp * abs(gmax - gmin))
    exp[case]['pt'] = pt
    exp[case]['peaks'] = scipy.signal.find_peaks(exp[case]['nov'], height=pt)
    exp[case]['peak_times'] = exp[case]['peaks'][0]*float(downsampling)*hop_size/sr



change_point_list = vocal_file_paths_and_change_points[index]["change_points"]

result = [item*sr/hop_size/downsampling for item in change_point_list]
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# all_peaks = exp['c3']["peaks"][0]*float(downsampling)*hop_size/sr
# print("Prediction times:", all_peaks)
for i, case in enumerate(['c1', 'c2', 'c3']):

    ax[0, i].matshow(exp[case]['rep'], cmap='Greys')
    matrix_shape = exp[case]['nov'].shape[0]
    x = np.arange(matrix_shape)*downsampling*hop_size/sr
    ax[1, i].plot(exp[case]['nov'], color='black')
    ax[1, i].set_xlim(0, exp[case]['nov'].shape[0])
    ax[1, i].axhline(exp[case]['pt'], color='black', ls=':', alpha=0.5)
    for peak in exp[case]["peaks"][0]:
        ax[1, i].axvline(peak, color='r', ls='--', alpha=0.5)
    for ann in result:
        ax[1, i].axvline(ann, color='b', ls='--', alpha=0.5)

fig.suptitle("Estimated structure of " + os.path.basename(audio_path), fontsize=18) 
plt.tight_layout()
plt.show()

distances = []
TP = 0
FP = 0
FN = 0

for case in ['c1','c2','c3']:
    distances = []
    TP = 0
    FP = 0
    FN = 0
    all_peaks = exp[case]["peaks"][0]*float(downsampling)*hop_size/sr
    for boundary in change_point_list:
        i = 0
        distances = []
        try:
            while boundary > all_peaks[i]:
                i += 1
        except:
            i -= 1
        # Take min distance between its neighbours now
        dist = min(abs(boundary - all_peaks[i]), abs(boundary - all_peaks[i-1]))
        distances.append(dist)
        if dist < 1:
            TP += 1
        else:
            FN = 1

    for boundary in all_peaks:
        i=0
        try:
            while boundary >= change_point_list[i]:
                i += 1
        except:
                i -= 1
        dist = min(abs(boundary-change_point_list[i]), abs(boundary-change_point_list[i-1]))
        distances.append(dist)
        if dist < 1:
            TP += 1
        else:
            FP += 1
    
    precision = TP / (TP + FP)
    recall = TP/(TP+FN)
    f1 = TP/(TP+0.5*(FP+FN))
    print("precision for case", case, "is", precision)
    print("recall for case", case, "is", recall)
    print("F1 score for case", case, "is", f1)