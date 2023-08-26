### Load modules
import os # module for interacting with files and folders
from audio_processing import *
from audio_display import *
from array_manipulation import *
from load_audio import *

def get_narr(fns=narrative_fns, mfcc_n=12, pad = default_pad):
    training_data = []
    X_list = []
    for clip_fn in narrative_fns:
        clip_data = process_audio(clip_fn,n_mfcc_list=[2,3,12],pad=pad)
        training_data.append(clip_data)
        data = clip_data["mfcc"][str(mfcc_n)]
        X = np.transpose(data)
        X_list.append(X)
    training_X = np.concatenate(X_list,axis=0)
    narr = np.transpose(training_X)
    return narr

def get_scaler(fns=narrative_fns,mfcc_n=12,pad = default_pad):
    data = get_narr(fns=fns,mfcc_n=mfcc_n,pad=pad)
    scaler = get_hor_scaler(data)
    return scaler