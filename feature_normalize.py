### Load modules
import os # module for interacting with files and folders
from audio_processing import *
from audio_display import *
from array_manipulation import *
from load_audio import *

def get_narr(narr_fns, mfcc_n=12, get_fourier=False, pad = default_pad):
    training_data = []
    X_list = []
    for clip_fn in narr_fns:
        clip_data = process_audio(clip_fn,n_mfcc_list=[mfcc_n],pad=pad)
        training_data.append(clip_data)
        if get_fourier:
            data = clip_data["fourier"]
        else:
            data = clip_data["mfcc"][str(mfcc_n)]
        X = np.transpose(data)
        X_list.append(X)
    training_X = np.concatenate(X_list,axis=0)
    narr = np.transpose(training_X)
    return narr

def get_scaler(narr_fns,mfcc_n=12,pad = default_pad,scaler_type='robust'):
    data = get_narr(narr_fns=narr_fns,mfcc_n=mfcc_n,pad=pad)
    scaler = get_hor_scaler(data,scaler_type=scaler_type)
    return scaler