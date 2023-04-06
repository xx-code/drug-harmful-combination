import pandas as pd
import numpy as np
import os

def work_in_data():
    pass

def load_dataset(path='', is_order=False):

    df = pd.read_csv(path)

    y = df['hospit'].values
    
    if not is_order:
        X = df.drop(['patient_id','hospit'], axis=1)
    else :
        X = df.drop(['hospit'], axis=1)

    X = X.to_numpy()

    return X, y

def split_dataset_train_test(test_size, shuffle=False):
    if os.path.isfile('data/poly_only_trans.csv') and (not os.path.isfile('data/train_set.csv') or not os.path.isfile('data/test_set.csv')) :
        df = pd.read_csv('data/poly_only_trans.csv')
        df = df.drop(['poly', 'timestamp'], axis=1)
        df[df['hospit'] > 0] = 1

        df = df.sort_values(by=['hospit']).reset_index(drop=True)

        n_test = int(df.shape[0]*test_size)

        idx_hospit = df.index[df['hospit'] == 1]
        portion_idx_hospit = len(idx_hospit)/df.shape[0]
        idx_no_hospit = df.index[df['hospit'] == 0]
        portion_idx_no_hospit = len(idx_no_hospit)/df.shape[0]

        idx_no_hospit = idx_no_hospit[:int(n_test*portion_idx_no_hospit)]
        idx_hospit = idx_hospit[:int(n_test*portion_idx_hospit)]

        
        df_test_hospit = df.iloc[idx_hospit]
        df_test_no_hospit = df.iloc[idx_no_hospit]

        df_test = df_test_hospit.append(df_test_no_hospit)

        df = df.drop(idx_hospit)
        df = df.drop(idx_no_hospit)

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
            df_test = df_test.sample(frac=1).reset_index(drop=True)

        df.to_csv('data/train_set.csv', index=False)
        df_test.to_csv('data/test_set.csv', index=False)

        del df
        del df_test

        print('done!')
        return 'data/train_set.csv', 'data/test_set.csv'
    return None, None