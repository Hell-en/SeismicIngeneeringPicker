import pandas as pd
import numpy as np
from src import segyrw
from src import dsp
from tensorflow.keras.utils import to_categorical
import Reader


class Preparer:

    """ Class for preparing income data and creating datasets"""


    @staticmethod
    def _gen_set(path, df_header_subset):
        x = segyrw.read_sgy_traces(path, df_header_subset['SOU_X'].astype(int).values) # CHANGE INSIDE
        y = df_header_subset['FB'].astype(int).values
    
        x = dsp.normalize_traces_by_std(x, 255, axis=1)
        x = dsp.normalize_traces(x, scale_type='std')
        x = np.expand_dims(x, axis=-1)
    
        heaviside = to_categorical(y+1, num_classes=x.shape[1])
        y_map = np.cumsum(heaviside, axis=1)
        heaviside = to_categorical(y-1, num_classes=x.shape[1])
        y_zeros = np.fliplr(np.cumsum(np.fliplr(heaviside), axis=1))
        y_pick = to_categorical(y, num_classes=x.shape[1])

        y_mask = np.stack((y_zeros, y_pick, y_map), axis=2)

        y_det = to_categorical(y_pick, num_classes=2)

        heaviside = to_categorical(y, num_classes=x.shape[1])
        y_map = np.cumsum(heaviside, axis=1)
        y_heavi = to_categorical(y_map, num_classes=2)
        return x, y_pick, y_det, y_mask, y_heavi
    

    def _create_df(self, train_percent): # train_percent - 0/0.2 или др объем Train data from all data
        df_header = pd.read_csv(self.path)

        n = round(len(df_header.index)*train_percent)
        df_header_subset = df_header.sample(n, random_state=37)

        all_inds = df_header.index.values
        train_inds = df_header_subset.index.values
        test_inds = np.setdiff1d(all_inds, train_inds)

        df_header_test = df_header.iloc[test_inds]
        df_header_test = df_header.iloc[test_inds].sample(random_state=37)

        return df_header_subset, df_header_test


    def generate_data(self):
        list_of_files = []
        part = 0.2 # 0 для исслед, когда обученную модель тестить on new data
        df_header_subset, df_header_test = self._create_df(part)
        print("check df  ", df_header_subset.columns) # to check

        Reader.get_file_names(list_of_files)
        print("check list_of_files  ", list_of_files) # to check

        x, y_pick, y_det, y_mask, y_heavi = [], [], [], [], []
        x_test, y_pick_test, y_det_test, y_mask_test, y_heavi_test = [], [], [], [], []


        for file in list_of_files:  #sgy
            num_file = Reader.get_source_num(file)
            print("num_file = ", num_file)
            # далее надо склеивать данные в один df
            ex, p, d, m, h = self._gen_set(file, df_header_subset[df_header_subset['SOU_X'] == num_file]) # df_header_subset[df_header_subset.columns[0] == num_file]
            x.append(ex)
            y_pick.append(p)
            y_det.append(d)
            y_mask.append(m)
            y_heavi.append(h)

            ex, p, d, m, h = self._gen_set(file, df_header_subset[df_header_subset['SOU_X'] == num_file])
            x_test.append(ex)
            y_pick_test.append(p)
            y_det_test.append(d)
            y_mask_test.append(m)
            y_heavi_test.append(h)

        return x, y_pick, y_mask, x_test
