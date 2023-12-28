import logging
import numpy as np
import torch
from torch.utils.data import  TensorDataset
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATASET_TASK = {'boston': 'regression',
                'breast': 'classification'}

class UCIDataset():

    def __init__(self, dataset_path, k=-1, standardize=True, seed=0):
        #dataset_path = ('./data/' + dataset + '.pth')
        logger.info('Loading dataset from %s' % dataset_path)
        dataset = TensorDataset(*torch.load(dataset_path))
        X, Y = dataset.tensors
        X, Y = X.numpy(), Y.numpy()

        if k !=-1 :
            assert k > 0
            self.kfold = KFold(n_splits=k)
            self.X_train_kfold = []
            self.Y_train_kfold = []
            self.X_test_kfold = []
            self.Y_test_kfold = []
            self.Y_train_mean_kfold = []
            self.Y_train_std_kfold = []
            for train_index, test_index in self.kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                self.X_train_kfold.append(torch.tensor(X_train, dtype=torch.float64))
                self.X_test_kfold.append(torch.tensor(X_test, dtype=torch.float64))
                # Standardize data
                Y_train_mean, Y_train_std = Y_train.mean(0), Y_train.std(0) + 1e-9
                if standardize:
                    Y_train = (Y_train - Y_train_mean) / Y_train_std
                    Y_test = (Y_test - Y_train_mean) / Y_train_std
                self.Y_train_kfold.append(torch.tensor(Y_train, dtype=torch.float64))
                self.Y_test_kfold.append(torch.tensor(Y_test, dtype=torch.float64))
                self.Y_train_mean_kfold.append(Y_train_mean)
                self.Y_train_std_kfold.append(Y_train_std)
        else:
            #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
            X_train_indices_boolean = np.random.choice([1, 0], size=X.shape[0], p=[0.8, 0.2])
            X_train_indices = np.where(X_train_indices_boolean == 1)[0]
            X_test_indices = np.where(X_train_indices_boolean == 0)[0]
            self.X_train = torch.tensor(X[X_train_indices], dtype=torch.float64)
            self.Y_train = torch.tensor(Y[X_train_indices], dtype=torch.float64)
            self.X_test = torch.tensor(X[X_test_indices], dtype=torch.float64)
            self.Y_test = torch.tensor(Y[X_test_indices], dtype=torch.float64)
            """
            Pd = None
            if pca != -1:
                X_train, Pd = apply_pca(X_train, pca) # fit_transform X_train
                X_test = X_test @ Pd # transform X_test
            """
            # standardize Y labels
            self.Y_train_mean, self.Y_train_std = self.Y_train.mean(0), self.Y_train.std(0) + 1e-9
            if standardize:
                self.Y_train = (self.Y_train - self.Y_train_mean) / self.Y_train_std
                self.Y_test = (self.Y_test - self.Y_train_mean) / self.Y_train_std
