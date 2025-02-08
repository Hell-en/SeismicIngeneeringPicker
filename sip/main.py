from Preprocessing import Preprocesser
from Picker import CNNPicker
from Controller import QualityController


def processes(): # for no model learning

    processor = Preprocesser.Preprocess(dir_path='../FBdata/2016doroga/sgy/', picks_path='../FBdata/doroga_picks.txt')
    data, idx = processor.sgy2npy()
    picker = CNNPicker.CNNModel(path_to_weights="../fb_picking_notebooks/models/model_weights_mount")
    model = picker.create_model(data)
    model.load_weights('../fb_picking_notebooks/models/model_weights_mount')
    predicted_values = picker.prediction(model, data)
