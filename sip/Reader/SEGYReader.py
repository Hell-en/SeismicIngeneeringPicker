import numpy as np
import segyio as sio
import pandas as pd
from tqdm import tqdm

from Data import SEGYData

LIST_OF_ATTRIBUTES = [sio.TraceField.FieldRecord,
                      sio.TraceField.SourceX,
                      sio.TraceField.SourceY,
                      sio.TraceField.SourceSurfaceElevation,
                      sio.TraceField.GroupX,
                      sio.TraceField.GroupY,
                      sio.TraceField.ReceiverGroupElevation,
                      sio.TraceField.offset]
NAME_OF_ATTRIBUTES = ['FFID',
                      'SOUX',
                      'SOUY',
                      'SES',
                      'RECX',
                      'RECY',
                      'RGE',
                      'OFFSET']
NUMBER_OF_ATTRIBUTES = len(NAME_OF_ATTRIBUTES)


class Reader:
    """Class for .sgy seismic data reading 

    :param path (str): path to file.

    """

    def __init__(self, path):
        self.path = path

    @staticmethod
    def _get_segy_file(path, **kwargs):
        if 'endian' in kwargs:
            segy_file = sio.open(path, ignore_geometry=True, endian=kwargs['endian'])
        else:
            segy_file = sio.open(path, ignore_geometry=True)
        return segy_file

    @staticmethod
    def _get_traces(segy_file):
        traces = np.array([segy_file.trace[i] for i in range(segy_file.tracecount)])
        return traces

    @staticmethod
    def _get_header(segy_file, use_tqdm=True):
        headers = np.zeros((segy_file.tracecount, NUMBER_OF_ATTRIBUTES))
        if use_tqdm:
            for i in tqdm(range(headers.shape[0])):
                for j in range(headers.shape[1]):
                    headers[i, j] = segy_file.header[i][LIST_OF_ATTRIBUTES[j]]
        else:
            for i in range(headers.shape[0]):
                for j in range(headers.shape[1]):
                    headers[i, j] = segy_file.header[i][LIST_OF_ATTRIBUTES[j]]
        dataframe = pd.DataFrame(data=headers, columns=NAME_OF_ATTRIBUTES)
        return dataframe

    @staticmethod
    def _get_dt(segy_file):
        dt = segy_file.samples[1]
        return dt

    @staticmethod
    def _parse_segy_file(segy_file, **kwargs):
        traces = SEGYReader._get_traces(segy_file)
        headers = SEGYReader._get_header(segy_file, kwargs['use_tqdm'])
        dt = SEGYReader._get_dt(segy_file)
        return SEGYData(traces=traces, headers=headers, dt=dt)

    def get_data(self, ):
        segy_file = SEGYReader._get_segy_file(self.path)
        data = SEGYReader._get_traces(segy_file)
        return data

    def read(self, **kwargs):
        segy_file = SEGYReader._get_segy_file(self.path, **kwargs)
        segy_data = SEGYReader._parse_segy_file(segy_file, **kwargs)
        return segy_data
