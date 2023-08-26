# load modules for plotting
import matplotlib as matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display # for displaying acoustic information

# load modules for doing math
import numpy as np



default_res = 3
default_win = 5
default_ran = 80



def plot_spectrogram(fourier,sr,res=default_res,title='Spectrogram',win_ms=default_win,range = default_ran):
    hop = 2**(9-res)
    spec = fourier
    fig, ax = plt.subplots()
    spec_dB = librosa.power_to_db(spec**2, ref=np.max)
    img = librosa.display.specshow(spec_dB, x_axis='time', y_axis='hz',cmap='gray_r',ax=ax,
                                   hop_length = hop, vmin=-range)
    fig.colorbar(img, ax=[ax])
    title = ax.set(title=title)

def plot_mfcc(mfcc,sr,title='MFCC',res=default_res,color='coolwarm'):
    fig, ax = plt.subplots()
    hop = 2**(9-res)
    img = librosa.display.specshow(mfcc, x_axis='time', y_axis='hz',ax=ax,cmap=color,
                                   hop_length = hop)
    fig.colorbar(img, ax=[ax])
    title = ax.set(title=title)

def plot_timeseries(data,sr,title='data',res=default_res):
    hop = 2**(9-res)
    data = np.squeeze(data)
    times = librosa.times_like(data,sr=sr,hop_length=hop)
    fig, ax = plt.subplots()
    ax.plot(times, data, color='black')
    ax.set(title=title)

def plot_timeseries_on_spectrogram(fourier,data,sr,scale_ts=0.5,res=default_res,title='data',win_ms=default_win,range = 100,line_color='white'):
    hop = 2**(9-res)
    spec = fourier
    data = np.squeeze(data)
    data_max = max(data)
    fig, ax = plt.subplots()
    spec_dB = librosa.power_to_db(spec**2, ref=np.max)
    img = librosa.display.specshow(spec_dB, x_axis='time', y_axis='hz',cmap='gray_r',ax=ax,
                                hop_length = hop, vmin=-range)
    times = librosa.times_like(data,sr=sr,hop_length=hop)
    #times = np.zeros(spec_dB.shape[1])
    ax.plot(times, data*scale_ts*10000/data_max, color=line_color)
    fig.colorbar(img, ax=[ax])
    title = ax.set(title=title)

def plot_timeseries_on_mfcc(mfcc,data,sr,scale_ts=0.5,title='MFCC',res=default_res,color='coolwarm',line_color='white'):
    fig, ax = plt.subplots()
    hop = 2**(9-res)
    data = np.squeeze(data)
    data_max = max(data)
    img = librosa.display.specshow(mfcc, x_axis='time', y_axis='hz',ax=ax,cmap=color,
                                   hop_length = hop)
    times = librosa.times_like(data,sr=sr,hop_length=hop)
    #times = np.zeros(spec_dB.shape[1])
    ax.plot(times, data*scale_ts*10000/data_max, color=line_color)
    fig.colorbar(img, ax=[ax])
    title = ax.set(title=title)

def show_the_spectrogram(audio_data,res=default_res,title='Spectrogram',win_ms=default_win,range = 80,size=(8,4)):
  fourier = audio_data["fourier"]
  sr = audio_data["sr"]
  plt.rcParams["figure.figsize"] = size
  plot_spectrogram(fourier,sr,res=res,title=title,win_ms=win_ms,range = default_ran)

def show_the_mfcc(audio_data,mfcc_n,res=default_res,size=(8,4)):
  mfcc = audio_data["mfcc"][str(mfcc_n)]
  sr = audio_data["sr"]
  title='MFCC'+str(mfcc_n)
  if mfcc_n == 1:
    return plot_timeseries(mfcc,sr*2**res,title=title)
  plot_mfcc(mfcc,sr,title,res=res)


def plot_sound_square(square,dur,sr,gap=0.25,title="Sound Square!",heat_max=400,heat_min=0,colors='gist_rainbow',size=4):
    scale = sr/64
    end = dur
    plt.rcParams["figure.figsize"] = (size,size)
    plt.imshow(square,
            cmap = colors,
            vmax = heat_max,
            vmin = heat_min,
            interpolation = 'nearest',
            aspect=1)
    plt.title(title)
    plt.colorbar()
    ticks = [x for x in np.arange(0,scale*end,scale*gap)]
    labels = [x for x in np.arange(0,end,gap)]
    plt.yticks(ticks=ticks,labels=labels)
    plt.xticks(ticks=ticks,labels=labels)
    plt.show()

def plot_sound_square_contour(square,dur,sr,gap=0.25,n_contours=10,title="Sound Square!",heat_max=400,heat_min=0,filled=False,colors='gist_rainbow',size=4):
  scale = sr/64
  end = dur
  plt.rcParams["figure.figsize"] = (size,size)
  if filled:
    plt.contourf(square,
                cmap = colors,
                levels = n_contours,
                vmax = heat_max,
                vmin = heat_min)
  else:
     plt.contour(square,
                cmap = colors,
                levels = n_contours,
                vmax = heat_max,
                vmin = heat_min)
  plt.title(title)
  plt.colorbar()
  ticks = [x for x in np.arange(0,scale*end,scale*gap)]
  labels = [x for x in np.arange(0,end,gap)]
  plt.yticks(ticks=ticks,labels=labels)
  plt.xticks(ticks=ticks,labels=labels)
  plt.show()


def plot_2d_path(data,title='data',size=(4,4),x_bounds = (-250,450),y_bounds=(-150,200)):
  data_x = data[0]
  data_y = data[1]
  fig, ax = plt.subplots()
  fig.set_size_inches(size)
  ax.plot(data_x, data_y, 'o', ls='-', ms=4, markevery=2, color='black')
  ax.plot(data_x, data_y, 'o', ls='', ms=4, markevery=slice(0,40,2), color='red')
  ax.plot(data_x, data_y, 'o', ls='', ms=10, markevery=[0], color='red')
  ax.set(xlim=x_bounds)
  ax.set(ylim=y_bounds)
  ax.set(title=title)