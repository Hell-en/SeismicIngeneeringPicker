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


    def get_file_names(self,):
        txtfiles = []
        for file in glob.glob("*.sgy"): # отсечь лишнее
            txtfiles.append(file)


    def get_sou_num(file_name):
      string = string[:-4]  # get rid of .sgy
      string = file_name
      num = 0
      string = re.sub(r'^.*?_', '', string) # delete before _
      string = string.replace(re.search(r'(?:_)(.*)', string).group(), '') ## delete afetr _

      if string[0] == '-':  #starts with -
          num =  -int(string[1:].lstrip('0')) ## then
      else:
          num = int(string.lstrip('0'))
      return num

    
