from array_manipulation import *
import numpy as np


def trunc_mfcc(mfcc,frame_shift,front=True):
  frame_list = split_frames(mfcc)
  if front:
    trunc_list = frame_list[frame_shift:]
  else:
    trunc_list = frame_list[:-frame_shift]
  new_mfcc = stack_frames(trunc_list)
  return new_mfcc

def get_mfcc_diff(mfcc,frame_shift):
  front_mfcc = trunc_mfcc(mfcc,frame_shift,front=True)
  back_mfcc = trunc_mfcc(mfcc,frame_shift,front=False)
  diff_mfcc = np.subtract(back_mfcc,front_mfcc)
  return diff_mfcc

def get_vels(mfcc,frame_shift,norm=L2):
  diff = get_mfcc_diff(mfcc,frame_shift)
  vels = get_norms(diff)
  return vels