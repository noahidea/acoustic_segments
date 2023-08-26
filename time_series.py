import tslearn
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.metrics import dtw,soft_dtw

def recurrence_score(frames,comp_frames_list,dist = 'dtw'):
    score = 0
    for comp_frames in comp_frames_list:
        if dist == 'dtw':
            d = dtw(frames,comp_frames)
        score += (d+0.01)**(-2) # switch to exp(-d/const)
    return score
