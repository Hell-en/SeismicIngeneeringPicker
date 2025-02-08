from dataclasses import dataclass

import pandas as pd
import numpy as np
from numpy import typing as npt


@dataclass
class SEGYData:
    """Класс для хранения segy данных.
    Attributes:
        traces (npt.NDArray[np.float_]): Сейсмические трассы.
        dt (float): Шаг дискретизации по времени в мс.
        headers (pd.DataFrame): Таблица с заголовками.
    """

    traces: npt.NDArray[np.float_]
    dt: np.float_
    headers: pd.DataFrame
