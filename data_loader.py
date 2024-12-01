import numpy as np
import torch
import pandas as pd
import os
import ssl
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

def gen_concrete(normalization):
    # fetch dataset
    path = os.path.join(os.path.dirname(__file__), 'datasets', 'concrete.csv')
    concrete = pd.read_csv(path, sep=',')

    x = concrete.drop(['No', 'SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)'], axis=1).to_numpy()
    y = concrete['Compressive Strength (28-day)(Mpa)'].to_numpy()

    if normalization == True:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    data_loader = TensorDataset(x_tensor, y_tensor)

    return x_tensor, y_tensor, data_loader

def gen_lenses(normalization):
    # fetch dataset
    path = os.path.join(os.path.dirname(__file__), 'datasets', 'lenses.csv')
    lenses = pd.read_csv(path, names = ['age', 'spectacle_prescription', 'astigmatic', 'tear_production', 'class'], sep = '  ')

    x = lenses[['age', 'spectacle_prescription', 'astigmatic', 'tear_production']].to_numpy()
    y = lenses[['class']].to_numpy()

    if normalization == True:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    data_loader = TensorDataset(x_tensor, y_tensor)

    return x_tensor, y_tensor, data_loader

def gen_auto(normalization):
    # fetch dataset
    path = os.path.join(os.path.dirname(__file__), 'datasets', 'auto.csv')
    auto = pd.read_csv(path, names = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"], na_values = '?')

    # replace missing num-doors value with the mode (four doors)
    auto["num-of-doors"].fillna(auto["num-of-doors"].mode()[0], inplace=True)

    # replace other missing values with mean
    auto["normalized-losses"].fillna(auto["normalized-losses"].astype("float").mean(), inplace=True)
    auto["bore"].fillna(auto["bore"].astype("float").mean(), inplace=True)
    auto["stroke"].fillna(auto["stroke"].astype("float").mean(), inplace = True)
    auto["peak-rpm"].fillna(auto["peak-rpm"].astype("float").mean(), inplace = True)
    auto['horsepower'].fillna(auto['horsepower'].astype("float").mean(), inplace=True)

    # drop rows missing price
    auto.dropna(subset=["price"], axis=0, inplace=True)
    auto.reset_index(drop=True, inplace=True)

    # convert types
    auto[["normalized-losses","horsepower"]] = auto[["normalized-losses","horsepower"]].astype("int")

    # get non object features
    x = auto.iloc[:, :-1].select_dtypes(exclude='object').to_numpy()
    y = auto.iloc[:,-1].to_numpy()

    print(x.shape)
    print(y.shape)

    if normalization == True:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    data_loader = TensorDataset(x_tensor, y_tensor)

    return x_tensor, y_tensor, data_loader