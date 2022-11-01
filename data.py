import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

class tabular_dataset(Dataset):
    def __init__(self, path, csv_prefix=None,
                npy_prefix=None, transform=None, minmax=True,
                tol=1e-20, noisy_prop=0.3, is_train=True):
        """
        Args: 
            path (string): datadir to find the csv files.
            csv_file (string): Prefix of the csv files, assuming it lives in 
                                datadir. To pass in 
                                Postures_X.csv, simply pass in 'Postures'
            transform (callable, optional): Optional transform to be applied
                on a sample.
            minmax (callable, optional): Optional minmax argument to transform
            the datasets, where each row is one data point. Notice that minmax
            is not for labels.  
            tol (callable, optional): Optional argument for tolerance in minmax
            scaling. 
        """
        if csv_prefix =='insurance':
            csv_file_x = path + csv_prefix + '_X.csv'
            csv_file_y = path + csv_prefix + '_Y.csv'
            self.features = np.array(pd.read_csv(csv_file_x))
            self.labels = np.array(pd.read_csv(csv_file_y)).reshape(-1)
            n_class = 2
        else:
            raise NotImplementedError("check csv_prefix")

        np.random.seed(1004)
        rand_ind = np.random.permutation(len(self.features))
        self.features = self.features[rand_ind]
        self.labels = self.labels[rand_ind]

        if is_train is True:
            self.features = self.features[:-5000]
            self.labels = self.labels[:-5000]
        else:
            self.features = self.features[-5000:]
            self.labels = self.labels[-5000:]

        self.transform = transform
        self.minmax = minmax

        if self.minmax: 
            self.features = self._minmax_scale(self.features, tol)

        if noisy_prop > 0.0:
            N = len(self.labels)
            N_noisy = int(N*noisy_prop)
            rnd_index = np.random.choice(N, N_noisy, replace=False)
            self.labels[rnd_index] = np.random.randint(0, n_class, [N_noisy])

    def __len__(self):
        return len(self.features)

    def _minmax_scale(self, X, tol):
        """ 
        Args: 
            X : a numpy array dataset to apply minmax scaling to. 
            tol: tolerance for numerical stability
        """
        numerator = X - np.min(X,axis=0)
        denominator = np.max(X,axis=0) - np.min(X,axis=0) + tol
        return numerator / denominator

    def __getitem__(self, idx):
        """ 
        Args:
            idx: scalar index at dataset (int). 
        Returns:
            A transformed sample for features and label.
            Features are shape (batch, n_features), elevation labels are 
            (batch,1).
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx]
        features = features.astype('float').reshape(-1)
        label = self.labels[idx].reshape(-1)
        sample = {'features': features, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['features'], sample['label']



def mnist(is_train=True, noisy_prop=0.0, path='data_path'):
    dataset = torchvision.datasets.MNIST(path, train=is_train, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    if noisy_prop > 0.0:
        N = len(dataset.targets)
        N_noisy = int(len(dataset.targets)*noisy_prop)
        rnd_index = np.random.choice(N, N_noisy, replace=False)
        dataset.targets[rnd_index] = torch.randint(0, 10, [N_noisy])

    return dataset

def fashion_mnist(is_train=True, noisy_prop=0.0, path='data_path'):
    dataset = torchvision.datasets.FashionMNIST(path, train=is_train, download=True,
                                                transform=torchvision.transforms.ToTensor())
    if noisy_prop > 0.0:
        N = len(dataset.targets)
        N_noisy = int(len(dataset.targets)*noisy_prop)
        rnd_index = np.random.choice(N, N_noisy, replace=False)
        dataset.targets[rnd_index] = torch.randint(0, 10, [N_noisy])

    return dataset

def cifar10(is_train=True, noisy_prop=0.0, path='data_path'):
    dataset = torchvision.datasets.CIFAR10(path, train=is_train, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                             (0.2023, 0.1994, 0.2010))
                                            ]))
    if noisy_prop > 0.0:
        N = len(dataset.targets)
        N_noisy = int(len(dataset.targets)*noisy_prop)
        rnd_index = np.random.choice(N, N_noisy, replace=False)
        rnd_class = np.random.randint(0, 10, [N])
        for j in rnd_index:
            dataset.targets[j] = rnd_class[j]

    return dataset


    