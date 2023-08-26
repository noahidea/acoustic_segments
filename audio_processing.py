# load modules for writing sound files
import soundfile as sf

# load modules from the librosa library for acoustic processing and analysis
import librosa
import librosa.display # for displaying acoustic information
import librosa.feature # for extracting and working with audio features

# load module for playing sound files within the notebook
from IPython.display import Audio, display


# load modules for doing math
import numpy as np


default_res = 3
default_bdw = 50
default_pad = 0.1
default_win = 5

#### INPUT ####

# get timeseries
def get_audio_data(fn, dur = 0, offset = 0, pre_emph = True):
  out_dict = {} # a dictionary to store the audio data
  # read the timeseries from a filename
  if dur == 0:
    timeseries, sample_rate = librosa.load(fn, offset = offset)
    # measure the duration of the timeseries
    dur = librosa.get_duration(y=timeseries, sr=sample_rate)
  else:
    timeseries, sample_rate = librosa.load(fn, duration = dur, offset = offset)
  # apply a pre-emphasis filter
  if pre_emph:
    timeseries = librosa.effects.preemphasis(timeseries)
  # store the audio data in the dictionary
  out_dict["fn"] = fn
  out_dict["start"] = offset
  out_dict["end"] = offset + dur
  out_dict["sr"] = sample_rate
  out_dict["ts"] = timeseries
  out_dict["dur"] = dur
  return out_dict

# pad a timeseries with silence
def pad_timeseries(ts, sr, front_pad, back_pad):
    front_n = round(front_pad*sr)
    back_n = round(back_pad*sr)
    front_silence = np.zeros((front_n))
    back_silence = np.zeros((back_n))
    ts = np.concatenate([front_silence,ts,back_silence])
    return ts

# get fourier data from a timeseries
def get_fourier(ts,sr,res,win_ms=default_win):
  win = round(sr*0.001*win_ms)
  hop = 2**(9-res)
  fourier = librosa.stft(ts,hop_length=hop,win_length=win)
  fourier = np.abs(fourier)
  return fourier

# get mfcc data from a timeseries
def get_mfcc(ts,sr,n_mfcc,res,bdw=default_bdw):
  hop = 2**(9-res)
  win = round(sr*0.001*bdw)
  mfcc = librosa.feature.mfcc(y=ts, n_mfcc = n_mfcc, hop_length = hop,n_fft = win)
  # mfcc = np.abs(mfcc)
  return mfcc

def get_mfcc_dict(ts,sr,n_mfcc_list,res,bdw=default_bdw):
  mfcc_dict = {}
  for n_mfcc in n_mfcc_list:
    mfcc = get_mfcc(ts,sr,n_mfcc,res,bdw=bdw)
    mfcc_dict[str(n_mfcc)] = mfcc
  return mfcc_dict

# get audio data, including fourier representation and mfcc features,
# from a sound file given the file name 'fn'
def process_audio(fn,
                  dur = 0, offset = 0, pre_emph = True,
                  pad = default_pad,
                  fourier = True, res = default_res, win_ms=default_win, bdw=default_bdw,
                  mfccs = True, n_mfcc_list = [12]):
  out_dict = get_audio_data(fn, dur=dur, offset=offset, pre_emph=True)
  ts = out_dict["ts"]
  sr = out_dict["sr"]
  if pad > 0:
    ts = pad_timeseries(ts, sr, front_pad=pad, back_pad=pad)
    out_dict["ts"] = ts
  if fourier:
    out_dict["fourier"] = get_fourier(ts, sr, res = res, win_ms = win_ms)
  if mfccs:
    out_dict["mfcc_ns"] = n_mfcc_list
    out_dict["mfcc"] = get_mfcc_dict(ts, sr, n_mfcc_list, res = res, bdw = bdw)
  return out_dict


#### OUTPUT ####

def export_audio_from_timeseries(fn, ts, sr, de_emph = True):
  # undo a preemphasis filter
  if de_emph:
    ts = librosa.effects.deemphasis(ts)
  sf.write(fn, ts, sr, subtype='PCM_24')

def timeseries_from_fourier(fourier,sr,res=default_res,win_ms=default_win):
  win = round(sr*0.001*win_ms)
  hop = 2**(9-res)
  ts = librosa.griffinlim(fourier,hop_length=hop,win_length = win)
  return ts

# using liftering, recover fourier data from mfcc data
def timeseries_from_mfcc(mfcc, res=default_res):
  hop = 2**(9-res)
  mel_from_mfcc = librosa.feature.inverse.mfcc_to_mel(mfcc)
  fourier_from_mel = librosa.feature.inverse.mel_to_stft(mel_from_mfcc)
  ts = librosa.griffinlim(fourier_from_mel,hop_length=hop)
  return ts

# export an audio file to filename 'fn' from the fourier data
def export_audio(fn, audio_data, format = 'ts',
                 res = default_res, win_ms = default_win,
                 n_mfcc = 0, de_emph = True):
  sr = audio_data['sr']
  if format == 'ts':
    ts = audio_data['ts']
  elif format == 'fourier':
    fourier = audio_data["fourier"]
    ts = timeseries_from_fourier(fourier,sr,res=res,win_ms=win_ms)
  elif format == 'mfcc':
    if n_mfcc == 0:
      n_mfcc = audio_data['mfcc_ns'][0]
    mfcc = audio_data['mfcc'][str(n_mfcc)]
    ts = timeseries_from_mfcc(mfcc,res=res)
  export_audio_from_timeseries(fn,ts,sr,de_emph = de_emph)