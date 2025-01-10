import numpy as np
import pylab as plt
import pandas as pd
import ipywidgets as ipw
import os
import sys
sys.path.append('../')
from Reader import SEGYReader


class Preprocess:

  @staticmethod
  def sgy2npy(dir_path = '../FBdata/2016doroga/sgy/'):

    filenames = os.listdir(dir_path)
    filenames

    gather_list = []
    header_list = []
    dt_list = []
    for name in filenames:
      print(name)
      if name != '.ipynb_checkpoints':
        segy_file = SEGYReader(os.path.join(dir_path, name)).read(use_tqdm=False, endian='lsb')
        gather_list.append(segy_file.traces)
        header_list.append(segy_file.headers)
        dt_list.append(segy_file.dt)

    df_merged = pd.concat(header_list, axis=0)
    df_merged = df_merged.reset_index(drop=True)
    gather_merged = np.vstack(gather_list)

    if not picks_exist:
      np.save('Doroga2016.npy', gather_merged)
    else:
      trace_idx = []
      picks_idx = []
      for i in range(picks.shape[0]):
        sx = picks[i, 0]
        rx = picks[i, 1]
        df_i = df_merged[(df_merged['SOUX'] == sx) * (df_merged['RECX'] == rx)]
        if len(df_i) != 0:
          idx = df_i.index[0]
          trace_idx.append(idx)
          picks_idx.append(i)
      trace_idx = np.asarray(trace_idx)
      picks_idx = np.asarray(picks_idx)

      gather_with_picks = gather_merged[trace_idx]
      picks_for_gather = np.int32(picks[picks_idx, 2] / dt_list[0])

      np.save('Poligon2013.npy', gather_with_picks)


@staticmethod
def plot_(idx):  # with picks
    fig = plt.figure(figsize=(16, 4))
    plt.plot(gather_with_picks[idx])
    plt.axvline(picks_for_gather[idx], c='k', ls='dashed')
    plt.show()
