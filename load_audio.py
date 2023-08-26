### Load modules
import os # module for interacting with files and folders
from audio_processing import *
from audio_display import *
from array_manipulation import *

### Load Audio Files
paths = [r'sound/American-English/Narrative',
        r'sound/American-English/Consonants',
        r'sound/American-English/Vowels',
        r'sound/Non-speech']



default_mfcc_ns = [5,12,36]
default_win_ms = 5
default_pad = 0.1
default_bdw = 50

def load_audio(paths_list = paths,pad = default_pad, mfcc_ns = default_mfcc_ns, win_ms = default_win_ms, bdw = default_bdw,nonspeech_dur=2):
    narrative_fns = []
    consonants_fns = []
    vowels_fns = []
    nonspeech_fns = []
    fns = [narrative_fns,consonants_fns,vowels_fns,nonspeech_fns]
    for i,path in enumerate(paths_list):
        for root, dirs, files in os.walk(path):
            for file in files:
                fns[i].append(os.path.join(root,file))
    narrative_data = [process_audio(fn,n_mfcc_list=mfcc_ns,pad=pad,win_ms=win_ms,bdw=bdw) for fn in narrative_fns]
    consonants_data = [process_audio(fn,n_mfcc_list=mfcc_ns,pad=pad,win_ms=win_ms,bdw=bdw) for fn in consonants_fns]
    vowels_data = [process_audio(fn,n_mfcc_list=mfcc_ns,pad=pad,win_ms=win_ms,bdw=bdw) for fn in vowels_fns]
    nonspeech_data = [process_audio(fn,dur=nonspeech_dur,n_mfcc_list=mfcc_ns,pad=pad,win_ms=win_ms,bdw=bdw) for fn in nonspeech_fns]
    audio_data = narrative_data,consonants_data,vowels_data,nonspeech_data
    return fns,audio_data