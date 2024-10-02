from Reader import SEGYReader
from Preparer import DataPreparer
from Picker import CNNPicker
from Controller import QualityController


reader = SEGYReader.Reader('D:\all\study\SeismicIngeneeringPicker\test_data\first_breaks.txt', 'D:\all\study\SeismicIngeneeringPicker\test_data')
