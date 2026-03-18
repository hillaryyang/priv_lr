"""
===============
data_loader.py
===============
Load and preprocess the three benchmark datasets

Each loader:
1. Reads raw CSV from the datasets directory
2. Selects relevant feature/target columns
3. Handles missing values if needed
4. Applies normalization
5. Returns reformatted features/target and DataLoader

Datasets (all sourced from the UCI ML Repository)
- Lenses: 24 instances, 4 features, target = contact lens class
- Concrete: 103 instances, 7 features, target = compressive strength
- Automobiles: 201 instances, 15 features, target = price
"""

import os
import warnings

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")

def _to_tensors(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, TensorDataset]:
    """Normalize data and reformat to tensors and TensorDataset dataloader

    Args:
        x, y: feature matrix and target vector

    Returns:
        x_tensor, y_tensor: Feature/target tensors
        dataset: TensorDataset wrapping (x_tensor, y_tensor)
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel() # reshape to 2D for scaling, then flatten back to 1D

    # convert to torch tensors, compile into TensorDataset
    x_tensor, y_tensor = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return x_tensor, y_tensor, TensorDataset(x_tensor, y_tensor)

def gen_concrete() -> tuple[torch.Tensor, torch.Tensor, TensorDataset]:
    """Load Concrete dataset, 103 instances with 7 input features 
    (cement, slag, fly ash, water, SP, coarse aggregate, fine aggregate)
    and Compressive Strength (MPa) target

    Returns:
        x_tensor, y_tensor: feature/target tensors
        dataset: TensorDataset wrapping (x_tensor, y_tensor)
    """
    path = os.path.join(os.path.dirname(__file__), "datasets", "concrete.csv") # assign dataset path
    concrete = pd.read_csv(path, sep=",") # read dataset

    x = concrete.drop(
        ["No", "SLUMP(cm)", "FLOW(cm)", "Compressive Strength (28-day)(Mpa)"], axis=1
    ).to_numpy() # drop the index column and two intermediate outputs (slump, flow)
    y = concrete["Compressive Strength (28-day)(Mpa)"].to_numpy()

    return _to_tensors(x, y)


def gen_lenses() -> tuple[torch.Tensor, torch.Tensor, TensorDataset]:
    """Load Lenses dataset, 24 instances with 4 features
    (age, spectacle prescription, astigmatism, tear production rate)
    and contact lens target

    Returns:
        x_tensor, y_tensor: feature/target tensors
        dataset: TensorDataset wrapping (x_tensor, y_tensor)
    """
    path = os.path.join(os.path.dirname(__file__), "datasets", "lenses.csv") # assign dataset path
    lenses = pd.read_csv(
        path,
        names=["age", "spectacle_prescription", "astigmatic", "tear_production", "class"],
        sep="  ",
    ) # read dataset and assign feature column names

    x = lenses[["age", "spectacle_prescription", "astigmatic", "tear_production"]].to_numpy()
    y = lenses[["class"]].to_numpy()

    return _to_tensors(x, y)


def gen_auto() -> tuple[torch.Tensor, torch.Tensor, TensorDataset]:
    """Load Automobiles dataset

    Preprocessing steps:
    1. Only numeric columns are retained as features; categorical columns are dropped
    2. For the "number of doors" feature, missing values are filled with the column mode;
       all other columns with missing values are filled with the column mean
    3. Rows missing the price target are removed entirely

    Returns:
        x_tensor, y_tensor: feature/target tensors
        dataset: TensorDataset wrapping (x_tensor, y_tensor)
    """
    path = os.path.join(os.path.dirname(__file__), "datasets", "auto.csv")
    col_names = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
        "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base",
        "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
        "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
        "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price",
    ]

    # read in dataset with '?' for missing values, assign column names
    auto = pd.read_csv(path, names=col_names, na_values="?")
    auto["num-of-doors"].fillna(
        auto["num-of-doors"].mode()[0], inplace=True
    ) # fill missing entries for "num-of-doors" with column mode

    # fill remaining missing values with column mean
    for col in ["normalized-losses", "bore", "stroke", "peak-rpm", "horsepower"]:
        auto[col].fillna(auto[col].astype("float").mean(), inplace=True)

    auto.dropna(subset=["price"], axis=0, inplace=True) # drop rows missing a price label
    auto.reset_index(drop=True, inplace=True)

    auto[["normalized-losses", "horsepower"]] = (
        auto[["normalized-losses", "horsepower"]].astype("int")
    ) # two columns can be cast to integers after filling in means

    x = auto.iloc[:, :-1].select_dtypes(exclude="object").to_numpy() # drop categorical columns 
    y = auto.iloc[:, -1].to_numpy()

    return _to_tensors(x, y)

# map dataset names to loaders
_LOADERS = {
    "concrete": gen_concrete,
    "lenses":   gen_lenses,
    "auto":     gen_auto,
}

# CLI argument for running a specific dataset
def parse_datasets(description: str) -> list[str]:
    """Parse -d CLI argument

    Args:
        description: argparse program description string

    Returns: list of dataset names (all datasets if -d is omitted)
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        choices=list(_LOADERS),
        default=None,
        help="Dataset to run (default: all)"
    )
    args = parser.parse_args()
    return [args.d] if args.d else list(_LOADERS)


def load_dataset(name: str) -> tuple[torch.Tensor, torch.Tensor, TensorDataset]:
    """Load dataset by name

    Args:
        name: Dataset identifier (concrete, lenses, or auto)

    Returns:
        x_tensor, y_tensor: feature/target tensors
        dataset: TensorDataset wrapping (x_tensor, y_tensor)

    Raises:
        ValueError: If dataset isn't recognized
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_LOADERS)}")
    return _LOADERS[name]()
