import logging
import numpy as np
import torch
from torch.utils.data import  TensorDataset
from sklearn.model_selection import KFold
import pandas as pd
from .normalize import normalize_data, zscore_normalization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def apply_pca(X, n_comp):
    N = X.shape[0]
    C = (1/N) * X.T @ X # N x N
    A, P = np.linalg.eigh(C) # C = PAPᵀ
    Pd = P[:, ::-1][:, 0:n_comp] # D x d
    Z = X @ Pd # N x d
    return Z, Pd

class SimulatedDataset():

    def __init__(self, dataset, k=-1, normalize=True, pca_latents=-1, seed=0):
        logger.info(f'Loading dataset {dataset}')
        data = pd.read_csv(f'data/simulated/{dataset}.csv')
        X, Y = data.iloc[:,0:-1].to_numpy(), data.iloc[:,-1].to_numpy().reshape(-1,1)

        if normalize:
            X = (X - X.mean(0)) / (X.std(0)+1e-9)

        if k !=-1 :
            assert k > 0
            self.kfold = KFold(n_splits=k, shuffle=True)
            self.X_train_kfold = []
            self.Y_train_kfold = []
            self.X_test_kfold = []
            self.Y_test_kfold = []
            self.Y_train_mean_kfold = []
            self.Y_train_std_kfold = []

            for train_index, test_index in self.kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                # Apply PCA
                self.Pd = None
                if pca_latents != -1:
                    X_train, self.Pd = apply_pca(X_train, pca_latents) # fit_transform X_train
                    X_test = X_test @ self.Pd # transform X_test

                # Normalize data 
                #X_train_mean, X_train_std = X_train.mean(0), X_train.std(0) + 1e-9
                Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
                Y_train =  (Y_train - Y_train_mean) / Y_train_std  
                Y_test =  (Y_test - Y_train_mean) / Y_train_std

                
                # Save data
                self.X_train_kfold.append(torch.tensor(X_train, dtype=torch.float64))
                self.X_test_kfold.append(torch.tensor(X_test, dtype=torch.float64))
                self.Y_train_kfold.append(torch.tensor(Y_train, dtype=torch.float64))
                self.Y_test_kfold.append(torch.tensor(Y_test, dtype=torch.float64))
                self.Y_train_mean_kfold.append(Y_train_mean)
                self.Y_train_std_kfold.append(Y_train_std)
        else:
            X_train_indices_boolean = np.random.choice([1, 0], size=X.shape[0], p=[0.8, 0.2])
            train_indices = np.where(X_train_indices_boolean == 1)[0]
            test_indices = np.where(X_train_indices_boolean == 0)[0]
            X_train, X_test = X[train_indices], X[test_indices]
            Y_train, Y_test = Y[train_indices], Y[test_indices]

            self.Y_train_mean, self.Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
            Y_train =  (Y_train - self.Y_train_mean) / self.Y_train_std  
            Y_test =  (Y_test - self.Y_train_mean) / self.Y_train_std

            # Save data
            self.X_train = torch.tensor(X_train, dtype=torch.float64)
            self.Y_train = torch.tensor(Y_train, dtype=torch.float64)
            self.X_test = torch.tensor(X_test, dtype=torch.float64)
            self.Y_test = torch.tensor(Y_test, dtype=torch.float64)
            """
            Pd = None
            if pca != -1:
                X_train, Pd = apply_pca(X_train, pca) # fit_transform X_train
                X_test = X_test @ Pd # transform X_test
            """

