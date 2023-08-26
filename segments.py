# load modules for doing math
import numpy as np
import copy
import numpy.linalg as LA

# load modules for plotting
import matplotlib as matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display # for displaying acoustic information

import warnings


default_res = 3
default_win = 5
default_ran = 80

def get_intervals(clustering,frame_w):
    intervals = []
    start_frame = -1
    curr_label = None
    for frame,label in enumerate(clustering+[None]):
        if label != curr_label:
            if start_frame >= 0:
                last_interval = {}
                end_frame = frame
                last_interval["fmin"] = start_frame
                last_interval["fmax"] = end_frame
                last_interval["tmin"] = start_frame*frame_w
                last_interval["tmax"] = end_frame*frame_w
                last_interval["text"] = str(curr_label)
                intervals.append(last_interval)
            start_frame = frame
            curr_label = label
    return intervals

def get_even_segments(frame_n,frame_w,seg_ms=0,n_segments=0,clip_last = True,text_method = 'index'):
    frame_t = frame_n*frame_w
    if n_segments != 0:
        clip_last = False
        if seg_ms == 0:
            seg_ms = 1000*frame_t / n_segments
        else:
            raise Exception('Cannot specify both number of segments and width of segments. One of seg_ms and n_segments must be 0')
    seg_frame_n = seg_ms/(1000*frame_w)
    if seg_frame_n < 1:
        warnings.warn('Segment width must be at least one frame. Setting segment width to 1 frame width.')
        seg_ms = frame_w*1000
    intervals = []
    start_time = 0
    clipped = False
    while start_time < frame_t:
        last_interval = {}
        end_time = start_time + seg_ms/1000
        if end_time > frame_t:
            end_time = frame_t
            clipped = True
        start_frame = int(np.round(start_time/frame_w))
        end_frame = int(np.round(end_time/frame_w))
        last_interval["fmin"] = start_frame
        last_interval["fmax"] = end_frame
        last_interval["tmin"] = start_frame*frame_w
        last_interval["tmax"] = end_frame*frame_w
        if text_method == 'index':
            last_interval["text"] = str(len(intervals))
        else:
            last_interval["text"] = "0"
        intervals.append(last_interval)
        start_time = end_time
    if clipped and not clip_last:
        intervals.pop()
    return intervals

def refine_segmentation(segmentation,frame_ind_list,text_method = 'index'):
    fine_list_list = []
    for segment in segmentation:
        splits = []
        for i,frame_ind in enumerate(frame_ind_list):
            if segment["fmax"] > frame_ind and segment["fmin"] < frame_ind:
                splits.append(frame_ind)
        coarse = copy.deepcopy(segment)
        if splits == []:
            fine_list = [coarse]
        else:
            fine_list = []
            splits.insert(0,coarse["fmin"])
            splits.append(coarse["fmax"])
            w = coarse["tmax"]/coarse["fmax"]
            text = coarse["text"]
            for i,fmin in enumerate(splits[:-1]):
                fine = {}
                fmax = splits[i+1]
                fine["fmin"] = fmin
                fine["fmax"] = fmax
                fine["tmin"] = w*fmin
                fine["tmax"] = w*fmax
                if text_method == 'index':
                    fine["text"] = text+str(i)
                if text_method == 'raw_index':
                    fine["text"] = str(i)
                else:
                    fine["text"] = text
                fine_list.append(fine)
        fine_list_list.append(fine_list)
    new_segmentation = [fine for fine_list in fine_list_list for fine in fine_list]
    return new_segmentation

def prune_segmentation(segmentation,seg_ind_list,text_method = 'join'):
    new_segmentation = copy.deepcopy(segmentation)
    for seg_ind in sorted(seg_ind_list, reverse=True):
        left = new_segmentation[seg_ind]
        right = new_segmentation[seg_ind+1]
        new_segmentation.remove(left)
        new_segmentation.remove(right)
        new = join_segments([left,right],text_method = text_method)
        new_segmentation.insert(seg_ind,new)
    return new_segmentation

def join_segments(consecutive_segments,text_method = 'join'):
    for i,segment in enumerate(consecutive_segments[1:]):
        prev_segment = consecutive_segments[i]
        if segment["fmin"] != prev_segment["fmax"]:
            raise Exception(f'Segments must be consecutive.')
    text_list = [segment["text"] for segment in consecutive_segments]
    if text_method == 'join':
        joined_text = ''.join(text_list)
    elif text_method =='first':
        joined_text = text_list[0]
    elif text_method == 'last':
        joined_text = text_list[-1]
    elif text_method == 'longest':
        len_list = [segment["fmax"]-segment["fmin"] for segment in consecutive_segments]
        longest_ind = len_list.index(max(len_list))
        joined_text = text_list[longest_ind]
    elif text_method == 'count':
        joined_text = str(len(text_list))
    joined_interval = {}
    joined_interval["fmin"] = consecutive_segments[0]["fmin"]
    joined_interval["fmax"] = consecutive_segments[-1]["fmax"]
    joined_interval["tmin"] = consecutive_segments[0]["tmin"]
    joined_interval["tmax"] = consecutive_segments[-1]["tmax"]
    joined_interval["text"] = joined_text
    return joined_interval

def get_bouquet(segmentation,method='ngrams',n_iter = [1],text_method = 'count'):
    if method=='ngrams':
        bouquet = []
        for n in n_iter:
            i = 0
            while i+n <= len(segmentation):
                segments_to_join = segmentation[i:i+n]
                joined_interval = join_segments(segments_to_join,text_method = text_method)
                bouquet.append(joined_interval)
                i += 1
    return bouquet

'''def split_frames(intervals,frames):
    if intervals[-1]["fmax"] != frames.shape[0]:
        raise Exception(f'Intervals and frames must be the same length. Intervals are {intervals[-1]["fmax"]} and frames are {frames.shape[0]}')
    out_intervals = copy.deepcopy(intervals)
    splits = [interval["fmin"] for interval in intervals[1:]]
    frame_segment_list = np.split(frames,splits)
    for interval,segment in zip(out_intervals,frame_segment_list):
        interval["frames"] = segment
    return out_intervals'''

def split_frames(intervals,frames):
    out_intervals = []
    for interval in intervals:
        start = interval["fmin"]
        end = interval["fmax"]
        frame_segment = np.split(frames,[start,end])[1]
        out_intervals.append(frame_segment)
    return out_intervals

def plot_segments_on_spectrogram(fourier,intervals,sr,res=default_res,title='data',win_ms=default_win,range = 100,line_color='white',complete=True):
    if complete and intervals[-1]["fmax"]-intervals[0]["fmin"] != fourier.shape[1]:
        raise Exception(f'Intervals and fourier data must be the same length. Intervals are {intervals[-1]["fmax"]-intervals[0]["fmin"]} and fourier data are {fourier.shape[1]}')
    hop = 2**(9-res)
    spec = fourier
    fig, ax = plt.subplots()
    bounds = [interval["tmin"] for interval in intervals]+[intervals[-1]["tmax"]]
    mids = [0.5*(start+end) for start,end in zip(bounds[:-1],bounds[1:])]
    labels = [interval["text"] for interval in intervals]
    spec_dB = librosa.power_to_db(spec**2, ref=np.max)
    img = librosa.display.specshow(spec_dB, x_axis='time', y_axis='hz',cmap='gray_r',ax=ax,
                                hop_length = hop, vmin=-range)
    #times = np.zeros(spec_dB.shape[1])
    ax.vlines(bounds, 0, 10000, color=line_color,linewidths=0.75)
    for i,(xi, text) in enumerate(zip(mids,labels)):
        ax.text(xi, 5000+0*(int(text)+1)*(-1)**int(text), text,
                ha="center", va="center", size=10,
                bbox=dict(boxstyle="circle,pad=0.3",
                      fc="white", ec="black", lw=2))
    fig.colorbar(img, ax=[ax])
    title = ax.set(title=title)

def plot_segments_on_mfcc(mfcc,intervals,sr,title='MFCC',res=default_res,color='coolwarm',line_color='white',complete=True):
    if complete and intervals[-1]["fmax"]-intervals[0]["fmin"] != mfcc.shape[1]:
        raise Exception(f'Intervals and MFCC data must be the same length. Intervals are {intervals[-1]["fmax"]-intervals[0]["fmin"]} and mfcc data are {mfcc.shape[1]}')
    fig, ax = plt.subplots()
    hop = 2**(9-res)
    bounds = [interval["tmin"] for interval in intervals]+[intervals[-1]["tmax"]]
    mids = [0.5*(start+end) for start,end in zip(bounds[:-1],bounds[1:])]
    labels = [interval["text"] for interval in intervals]
    img = librosa.display.specshow(mfcc, x_axis='time', y_axis='hz',ax=ax,cmap=color,
                                   hop_length = hop)
    #times = np.zeros(spec_dB.shape[1])
    ax.vlines(bounds, 0, 10000, color=line_color,linewidths=0.75)
    for i,(xi, text) in enumerate(zip(mids,labels)):
        ax.text(xi, 5000+0*(int(text)+1)*(-1)**int(text), text,
                ha="center", va="center", size=10,
                bbox=dict(boxstyle="circle,pad=0.3",
                      fc="white", ec="black", lw=2))
    fig.colorbar(img, ax=[ax])
    title = ax.set(title=title)

def relabel_segments(labels,intervals):
    new_segmentation = []
    curr_label = None
    for interval,new_label in zip(intervals,labels):
        if new_label == curr_label:
            new_segmentation[-1]["fmax"] = interval["fmax"]
            new_segmentation[-1]["tmax"] = interval["tmax"]
        else:
            curr_label = new_label
            new_interval = copy.deepcopy(interval)
            new_interval["text"] = new_label
            new_segmentation.append(new_interval)
    return new_segmentation
