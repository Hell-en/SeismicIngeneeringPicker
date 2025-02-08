from Preprocessing import Preprocesser
from Picker import CNNPicker
from Controller import QualityController


def processes():

    data = Preprocesser(dir_path = '../FBdata/2016doroga/sgy/', picks_path = '../FBdata/doroga_picks.txt').sgy2npy()

