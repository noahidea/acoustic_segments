from time_series import *
from segments import *
import numpy as np
from scipy.signal import argrelextrema


def get_split_scores(frames,comp_frames_list,frame_n_list=None):
    scores = []
    if frame_n_list == None:
        frame_n_list = range(1,len(frames))
    for x in frame_n_list:
        left,right = np.split(frames,[x])
        s = 0.5*recurrence_score(left,comp_frames_list)+recurrence_score(right,comp_frames_list)
        scores.append(s)
    return scores

def get_best_splits(scores,threshhold = 0):
    best_splits, = argrelextrema(np.array(scores), np.greater)
    best_splits = list(best_splits)
    if 0 in best_splits:
        best_splits.remove(0)
    if len(scores)+1 in best_splits:
        best_splits.remove(len(scores)+1)
    best_splits = [(split,scores[split]) for split in best_splits]
    best_scores = [split[1] for split in best_splits if split[1] >= threshhold]
    best_splits = [split[0] for split in best_splits if split[1] >= threshhold]
    return best_splits,best_scores

def recurrence_prune(segmentation,frames,comp_frames_list,zeal=1,prune_adjacent=False,text_method = 'join',show_progress=False):
    old_segmentation = None
    new_segmentation = segmentation
    prune_round = 0
    if text_method == 'count':
        for segment in new_segmentation:
            segment["text"] = "1"
    while old_segmentation != new_segmentation:
        old_segmentation = new_segmentation
        segment_frames_list = split_frames(old_segmentation,frames)
        paired_segments = get_bouquet(old_segmentation,n_iter=[2],text_method='first')
        paired_segment_frames_list = split_frames(paired_segments,frames)
        prune_list = []
        score_list = []
        unjoined_scores = [recurrence_score(interval,comp_frames_list) for interval in segment_frames_list]
        joined_scores = [recurrence_score(interval,comp_frames_list) for interval in paired_segment_frames_list]
        for segment_ind,segment in enumerate(old_segmentation[:-1]):
            left_score = unjoined_scores[segment_ind]
            right_score = unjoined_scores[segment_ind+1]
            joined_score = joined_scores[segment_ind]
            if 0.5*(left_score+right_score) <= zeal*joined_score:
                prune_list.append(segment_ind)
                score_list.append(joined_score)
        smaller_ind_list = []
        if not prune_adjacent:
            for i,segment_ind in enumerate(prune_list[:-1]):
                if prune_list[i+1] == segment_ind+1:
                    j = score_list[i]
                    k = score_list[i+1]
                    if j <= k:
                        smaller_ind = i
                        larger_ind = i+1
                    if j > k:
                        smaller_ind = i+1
                        larger_ind = i
                    if smaller_ind not in smaller_ind_list:
                        smaller_ind_list.append(smaller_ind)
        prune_list = [ind for i,ind in enumerate(prune_list) if i not in smaller_ind_list]
        score_list = [score for i,score in enumerate(score_list) if i not in smaller_ind_list]
        new_segmentation = prune_segmentation(old_segmentation,prune_list,text_method=text_method)
        if show_progress:
            prune_round += 1
            print(f'Round {prune_round} of pruning complete.')
    return new_segmentation

def recurrence_refine(segmentation,frames,comp_frames_list,text_method = 'index',zeal=1,return_terminal = False,skip_segments = []):
    segment_frames_list = split_frames(segmentation,frames)
    terminal = []
    segments_and_frames = [x for i,x in enumerate(zip(segmentation,segment_frames_list)) if i not in skip_segments]
    split_list = []
    for segment,segment_frames in segments_and_frames:
        segment_score = recurrence_score(segment_frames,comp_frames_list)
        scores = get_split_scores(segment_frames,comp_frames_list)
        best_splits,best_scores = get_best_splits(scores,threshhold = segment_score/zeal)
        if best_splits == []:
            terminal.append(segment)
        split_list.extend([split+segment['fmin'] for split in best_splits])
    refined = refine_segmentation(segmentation,split_list,text_method=text_method)
    if return_terminal:
        return refined,terminal
    else:
        return refined