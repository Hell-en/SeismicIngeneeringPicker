import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical

from src import segyrw
from src import dsp
from src import plot_seismic
from src import plot_maps


class Reader:

    """Class for reading input files: .sgy and .txt with values of FB"""

    def __init__(self, path, pick):
        self.path = path
        self.pick = pick

    def _create_df(self):
        df_header = pd.read_csv(self.path)

        n = round(len(df_header.index)*0.2)
        df_header_subset = df_header.sample(n, random_state=37)
        df_header_subset['FB_NTC'] = np.int32(df_header_subset['FB']*500)

        all_inds = df_header.index.values
        train_inds = df_header_subset.index.values
        test_inds = np.setdiff1d(all_inds, train_inds)

        df_header_test = df_header.iloc[test_inds]
        df_header_test = df_header.iloc[test_inds].sample(random_state=37) # changed. check if ok
        df_header_test['FB_NTC'] = df_header_test['FB']*500

        return df_header_subset, df_header_test
    

    @staticmethod
    def _gen_set(path, df_header_subset):
        x = segyrw.read_sgy_traces(path, df_header_subset['IDX'].astype(int).values)
        y = df_header_subset['FB_NTC'].astype(int).values
    
        x = dsp.normalize_traces_by_std(x, 255, axis=1)
        x = dsp.normalize_traces(x, scale_type='std')
        x = np.expand_dims(x, axis=-1)
    
        heaviside = to_categorical(y+1, num_classes=x.shape[1])
        y_map = np.cumsum(heaviside, axis=1)
        heaviside = to_categorical(y-1, num_classes=x.shape[1])
        y_zeros = np.fliplr(np.cumsum(np.fliplr(heaviside), axis=1))
        y_pick = to_categorical(y, num_classes=x.shape[1])

        ### mask
        y_mask = np.stack((y_zeros, y_pick, y_map), axis=2)

        ### det
        y_det = to_categorical(y_pick, num_classes=2)

        # ### heavi
        heaviside = to_categorical(y, num_classes=x.shape[1])
        y_map = np.cumsum(heaviside, axis=1)
        y_heavi = to_categorical(y_map, num_classes=2)
        return x, y_pick, y_det, y_mask, y_heavi


    def generate_data(self):
        df_header_subset, df_header_test = Reader.create_df()
        x, y_pick, y_det, y_mask, y_heavi = Reader._gen_set(self.path, df_header_subset)
        x_test, y_pick_test, y_det_test, y_mask_test, y_heavi_test = Reader._gen_set(self.path, df_header_test)
        return 0 # what to return? what we need?
