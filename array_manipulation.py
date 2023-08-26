# load modules for doing math
import numpy as np
import numpy.linalg as LA
from sklearn import preprocessing
from sklearn.preprocessing import normalize

def split_frames(array):
  frame_n = array.shape[1]
  frame_list = np.split(array,frame_n,axis=1)
  return frame_list

def stack_frames(frame_list):
  array = np.concatenate(frame_list,axis=1)
  return array

def L2(v):
  norm = LA.norm(v,ord=2)
  return(norm)

def L1(v):
  norm = LA.norm(v,ord=1)
  return(norm)

def Ln1(v):
  norm = LA.norm(v,ord=-1)
  return(norm)

def L2_lim(v,lo,hi):
  norm = LA.norm(v,ord=2)
  if norm > hi:
    return hi
  elif norm < lo:
    return lo
  else:
    return(norm)

def get_norms(array,norm=L2):
  frames = split_frames(array)
  norms= [norm(v) for v in frames]
  norm_array = np.array(norms)
  return norm_array


def norm_array_vert(array,method='l2'):
  normed_array = normalize(np.abs(array), norm = method, axis=0)
  return normed_array

def norm_array_hor(array,scaler):
  yarra = np.transpose(array)
  normed_yarra = scaler.transform(yarra)
  normed_array = np.transpose(normed_yarra)
  return normed_array

def get_hor_scaler(array,scaler_type='standard'):
  yarra = np.transpose(array)
  if scaler_type == 'standard':
    scaler = preprocessing.StandardScaler().fit(yarra)
  elif scaler_type == 'minmax':
    scaler = preprocessing.MinMaxScaler().fit(yarra)
  elif scaler_type == 'maxabs':
    scaler = preprocessing.MaxAbsScaler().fit(yarra)
  elif scaler_type == 'robust':
    scaler = preprocessing.RobustScaler().fit(yarra)
  elif scaler_type == 'power':
    scaler = preprocessing.PowerTransformer().fit(yarra)
  elif scaler_type == 'quantile':
    scaler = preprocessing.QuantileTransformer().fit(yarra)
  elif scaler_type == 'gaussian':
    scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(yarra)
  elif scaler_type == 'normal':
    scaler = preprocessing.Normalizer().fit(yarra)
  return scaler