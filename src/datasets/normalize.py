import numpy as np

def zscore_normalization(X, mean=None, std=None, eps=1e-10):
    """Apply z-score normalization on a given data.

    Args:
        X: numpy array, shape [batchsize, num_dims], the input dataset.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        tuple: the normalized dataset and the resulting mean and variance.
    """
    if X is None:
        return None, None, None

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / (std + eps)

    return X_normalized, mean, std

def normalize_data(X_train, y_train, X_test=None, y_test=None):
    """Wrapper function used to normalize regression datasets.

    Args:
        X_train: np.array, [n_data, n_dims], the inputs of training data.
        y_train: np.array, [n_data, 1], the targets of training data.
        X_test: np.array, [n_data, n_dims], the inputs of test data.
        y_test: np.array, [n_data, 1], the targets of test data.
    """
    # Normalize the dataset
    X_train_, X_mean, X_std = zscore_normalization(X_train)
    y_train_, y_mean, y_std = zscore_normalization(y_train)

    if (X_test is not None) and (y_test is not None):
        X_test_, _, _ = zscore_normalization(X_test, X_mean, X_std)
        y_test_, _, _ = zscore_normalization(y_test)
        return X_train_, y_train_, X_test_, y_test_, y_mean, y_std
    else:
        return X_train_, y_train_, y_mean, y_std