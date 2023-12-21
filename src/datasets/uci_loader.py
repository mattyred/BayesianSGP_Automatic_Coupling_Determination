import logging
import numpy as np
import torch
from torch.utils.data import  TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class UCIDataset():

    def __init__(self, dataset_path, static_split, seed):
        #dataset_path = ('./data/' + dataset + '.pth')
        logger.info('Loading dataset from %s' % dataset_path)
        dataset = TensorDataset(*torch.load(dataset_path))
        X, Y = dataset.tensors
        X, Y = X.numpy(), Y.numpy()

        if static_split == False:
            Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-9
            return X, Y, Y_mean, Y_std
        else:
            #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=fold)
            X_train_indices_boolean = np.random.choice([1, 0], size=X.shape[0], p=[0.8, 0.2])
            X_train_indices = np.where(X_train_indices_boolean == 1)[0]
            X_test_indices = np.where(X_train_indices_boolean == 0)[0]
            self.X_train = torch.tensor(X[X_train_indices], dtype=torch.float32)
            self.Y_train = torch.tensor(Y[X_train_indices], dtype=torch.float32)
            self.X_test = torch.tensor(X[X_test_indices], dtype=torch.float32)
            self.Y_test = torch.tensor(Y[X_test_indices], dtype=torch.float32)
            """
            Pd = None
            if pca != -1:
                X_train, Pd = apply_pca(X_train, pca) # fit_transform X_train
                X_test = X_test @ Pd # transform X_test
            """
            # standardize Y labels
            self.Y_train_mean, self.Y_train_std = self.Y_train.mean(0), self.Y_train.std(0) + 1e-9
            self.Y_train = (self.Y_train - self.Y_train_mean) / self.Y_train_std
            self.Y_test = (self.Y_test - self.Y_train_mean) / self.Y_train_std
