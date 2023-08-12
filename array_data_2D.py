import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image


def array_para_data(filepath, num):
    df = pd.read_csv(filepath)
    para_training_data = []
    print(df.shape)
    for i in range(0, df.shape[0]):
        para_set = []
        for list_num in range(0, num):
            parameter = (df.iat[i, list_num])
            para_set.append(parameter)
        para_training_data.append(para_set)
    para_training_data = np.array(para_training_data)
    return (para_training_data)
