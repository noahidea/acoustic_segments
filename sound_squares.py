from array_manipulation import *
import numpy as np

def sound_square(mfcc,norm=L2):
  frames = split_frames(mfcc)
  square_list = []
  for v in frames:
    row_list = []
    for w in frames:
      d = np.subtract(v,w)
      n = norm(d)
      row_list.append(n)
    row = np.array(row_list)
    square_list.append(row)
  square = np.stack(square_list,axis=1)
  return square


def sound_rect(mfcc1,mfcc2,norm=L2):
  frames1 = split_frames(mfcc1)
  frames2 = split_frames(mfcc2)
  square_list = []
  for v in frames1:
    row_list = []
    for w in frames2:
      d = np.subtract(v,w)
      n = norm(d)
      row_list.append(n)
    row = np.array(row_list)
    square_list.append(row)
  square = np.stack(square_list,axis=1)
  return square