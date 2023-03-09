import pandas as pd
import numpy as np

def load_dataset(path=''):

    df = pd.read_csv(path)

    y = df['hospit'].values

    X = df.drop(['patient_id','hospit'], axis=1)
    X = X.to_numpy()

    return X, y