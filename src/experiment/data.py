"""
Helpers to load (and pre-process?) the ACIC 2018 data
dataset description: https://www.researchgate.net/publication/11523952_Infant_Mortality_Statistics_from_the_1999_Period_Linked_BirthInfant_Death_Data_Set
"""
import os
import shutil
import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_and_format_covariates(file_path='~/ml/IBM-Causal-Inference-Benchmarking-Framework/data/LBIDD/x.csv'):
    df = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')
    return df


def load_treatment_and_outcome(covariates, file_path, standardize=True):
    output = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')

    dataset = covariates.join(output, how='inner')
    t = dataset['z'].values
    y = dataset['y'].values
    x = dataset.values[:, :-2]
    if standardize:
        normal_scalar = preprocessing.StandardScaler()
        x = normal_scalar.fit_transform(x)
    return t.reshape(-1, 1), y.reshape(-1, 1), dataset.index, x


def load_ufids(file_path='/Users/claudiashi/data/small_sweep/params.csv'):
    df = pd.read_csv(file_path, header=0, sep=',')
    df = df[df['size'] > 4999]
    df = df[df['size'] < 10001]
    dfs = []
    for i in range(1, 64):
        dfs.append(df[df['dgp'] == i].sample(n=1))
    df = pd.concat(dfs, sort=False)
    ufids = df['ufid'].values
    return ufids


def make_subdirs(ufids, path):
    p = os.path.join(path, 'a')
    os.makedirs(p, exist_ok=True)
    for ufid in ufids:
        file = os.path.join(path, 'scaling/factuals/', ufid + '.csv')
        shutil.copy(file, p)


def load_params(file_path='/Users/claudiashi/data/small_sweep/params.csv'):
    df = pd.read_csv(file_path, header=0, sep=',')
    df = df[df['size'] > 4999]
    df = df[df['size'] < 10001]
    return df


def main():
    data_path = '../../dat/LIBDD/'
    simulation_id = '43b75dcfc0fc49beb95a111098ae11b1'

    ufids = load_ufids(os.path.join(data_path, 'scaling/params.csv'))
    make_subdirs(ufids, data_path)

    covariate_path = os.path.join(data_path, 'x.csv')
    all_covariates = load_and_format_covariates(covariate_path)
    simulation_file = os.path.join(data_path, 'scaling/factuals/', simulation_id + '.csv')
    t, y, sample_id, x = load_treatment_and_outcome(all_covariates, simulation_file)


if __name__ == '__main__':
    main()
    pass
