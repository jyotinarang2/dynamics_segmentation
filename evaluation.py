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

def predict_boundaries(results, paths, hop_size, downsampling, sr):
    predictions = {}
    invalid_songs = []

# some songs seem to be too short (need to determine how short) for the analysis
# they seem to almost entirely be speech instead of music, so they won't have annotations anyway
    for path in paths:
        p_name = os.path.basename(path)
        print(p_name)
        predictions[p_name] = {}

        for l in ["9", "12", "15", "18", "21"]:
            predictions[p_name][l] = {}

            for ptp in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                predictions[p_name][l][str(ptp)] = {}

                for case in ['L', 'a1', 'a2']:

                    gmax = np.amax(results[p_name][l][case])
                    # check for invalid pair (will be all zeros)
                    try:
                        gmin = np.amin(results[p_name][l][case][np.nonzero(results[p_name][l][case])])
                    except:
                        invalid_songs.append(p_name)

                    pt = gmin + (ptp * abs(gmax - gmin))
                    
                    # get peaks over threshold
                    pred = scipy.signal.find_peaks(results[p_name][l][case], height=pt)[0]
                    pred_time = pred*float(downsampling)*hop_size/sr

                    # # add boundary for 0 and END
                    # pred_time = np.insert(pred_time, 0, 0)
                    # pred_time = np.append(pred_time, results[p_name][l][case].shape[0])

                    # scale boundaries to time index of original song (2fps to 1fps)
                    #pred = pred / 2

                    predictions[p_name][l][str(ptp)][case] = pred_time
    return predictions

#Evaulate the change points
def evaluate_closeness_of_change_points_to_predictions(predictions, results, change_point_list_of_songs):
    dev = {}
    hit = {}
    for song in list(results.keys()):
        song_name = os.path.basename(song)
        if os.path.basename(song) not in predictions:
            continue
        dev[song] = {}
        hit[song] = {}
        for l in ["9", "12", "15", "18", "21"]:
            dev[song][l] = {}
            hit[song][l] = {}

            for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
                dev[song][l][ptp] = {}
                hit[song][l][ptp] = {}
                
                for case in ['L', 'a1', 'a2']:
                    dev[song][l][ptp][case] = {}
                    hit[song][l][ptp][case] = {}

                    mean_dev = []
                    precision = []
                    recall = []
                    f1 = []
                    for ann in change_point_list_of_songs:
                
                        distances = []
                        TP = 0
                        FP = 0
                        FN = 0

                        for boundary in change_point_list_of_songs[ann]:
                            i=0
                            try:
                                while boundary > predictions[song_name][l][ptp][case][i]:
                                    i += 1
                            except:
                                i -= 1
                            # Take min distance between its neighbours now
                            dist = min(abs(boundary - predictions[song_name][l][ptp][case][i]), abs(boundary - predictions[song_name][l][ptp][case][i-1]))
                            distances.append(dist)
                            if dist < 1:
                                TP += 1
                            else:
                                FN = 1
                        for boundary in predictions[song_name][l][ptp][case]:
                            i=0
                            try:
                                while boundary >= ann:
                                    i += 1
                            except:
                                    i -= 1
                            dist = min(abs(boundary-change_point_list_of_songs[ann][i]), abs(boundary-change_point_list_of_songs[ann][i-1]))
                            distances.append(dist)
                            if dist < 1:
                                TP += 1
                            else:
                                FP += 1
                        mean_dev.append(median(distances))
                        precision.append(TP / (TP + FP))
                        recall.append(TP / (TP + FN))
                        f1.append(TP / (TP + 0.5*(FP + FN)))
                                    # save median
                    dev[song][l][ptp][case] = mean(mean_dev)
                    # save scores
                    hit[song][l][ptp][case]["precision"] = mean(precision)
                    hit[song][l][ptp][case]["recall"] = mean(recall)
                    hit[song][l][ptp][case]["f1"] = mean(f1)
    return dev, hit

#Evaluate each song individually
# def compute_best_parameter_for_song(song):
#     for l in ["9", "12", "15", "18", "21"]:
#         for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
#             for case in ['L', 'a1', 'a2']:

#Compute the best case for the individual points   
def compute_best_case_for_all_parameters(dev, hit, results, predictions):
    dev_min = 10000
    dev_best = ""

    for l in ["9", "12", "15", "18", "21"]:
        for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
            for case in ['L', 'a1', 'a2']:
                devs = []
                for song in list(results.keys()):
                    if song not in predictions:
                        continue
                    devs.append(dev[song][l][ptp][case])
                if mean(devs) < dev_min:
                    dev_min = mean(devs)
                    dev_best = f"l: {l} | ptp: {ptp} | rep: {case}"
    print("Best configuration:", dev_best, "with median deviation", dev_min)

def compute_best_f1_for_all_parameters(dev, hit, results, predictions):
    f1_max = 0
    f1_best = ""

    for l in ["9", "12", "15", "18", "21"]:
        for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
            for case in ['L', 'a1', 'a2']:
                f1s = []
                for song in list(results.keys()):
                    if song not in predictions:
                        continue
                    f1s.append(hit[song][l][ptp][case]["f1"])
                if mean(f1s) > f1_max:
                    f1_max = mean(f1s)
                    f1_best = f"l: {l} | ptp: {ptp} | rep: {case}"
    print("Best configuration:", f1_best, "with F-measure", f1_max)

def compute_best_precision_for_all_parameters(dev, hit, results, predictions):
    # get highest precision
    precision_max = 0
    precision_best = ""

    for l in ["9", "12", "15", "18", "21"]:
        for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
            for case in ['L', 'a1', 'a2']:
                precisions = []
                for song in list(results.keys()):
                    if song not in predictions:
                        continue
                    precisions.append(hit[song][l][ptp][case]["precision"])
                if mean(precisions) > precision_max:
                    precision_max = mean(precisions)
                    precision_best = f"l: {l} | ptp: {ptp} | rep: {case}"
    print("Best configuration:", precision_best, "with precision", precision_max)

def compute_best_recall_for_all_parameters(dev, hit, results, predictions):
    # get highest recall
    recall_max = 0
    recall_best = ""

    for l in ["9", "12", "15", "18", "21"]:
        for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
            for case in ['L', 'a1', 'a2']:
                recalls = []
                for song in list(results.keys()):
                    if song not in predictions:
                        continue
                    recalls.append(hit[song][l][ptp][case]["recall"])
                if mean(recalls) > recall_max:
                    recall_max = mean(recalls)
                    recall_best = f"l: {l} | ptp: {ptp} | rep: {case}"
    print("Best configuration:", recall_best, "with recall", recall_max)    


def compute_best_case_precision_for_each_song(dev, hit, results, predictions):
    precisions = []
    configuration = []
 
    for song in list(results.keys()):
        for l in ["9", "12", "15", "18", "21"]:
            for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
                for case in ['L', 'a1', 'a2']:
                    precisions.append(hit[song][l][ptp][case]["precision"])
                    configuration.append((l, ptp, case))
                    
        print("For song:", song, "best precision is", max(precisions), configuration[np.argmax(precisions)])


def compute_best_case_recall_for_each_song(dev, hit, results, predictions):
    recalls = []
    configuration = []
 
    for song in list(results.keys()):
        for l in ["9", "12", "15", "18", "21"]:
            for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
                for case in ['L', 'a1', 'a2']:
                    recalls.append(hit[song][l][ptp][case]["recall"])
                    configuration.append((l, ptp, case))
                    
        print("For song:", song, "best recall is", max(recalls), configuration[np.argmax(recalls)])

def compute_best_case_f1_for_each_song(dev, hit, results, predictions):
    f1 = []
    configuration = []

    for song in list(results.keys()):
        for l in ["9", "12", "15", "18", "21"]:
            for ptp in ["0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8"]:
                for case in ['L', 'a1', 'a2']:
                    f1.append(hit[song][l][ptp][case]["f1"])
                    configuration.append((l, ptp, case))
                    
        print("For song:", song, "best f1 score is", max(f1), configuration[np.argmax(f1)])


