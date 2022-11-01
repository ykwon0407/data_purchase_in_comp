import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils import data 
from tqdm import tqdm
import random
import numpy as np
import numpy.random as np_rand
from torch.utils.data import DataLoader
from data import *

class SLAgent(nn.Module):
    def __init__(self, X=None, y=None, n_class=10):
        super(SLAgent, self).__init__()
        self.X = X
        self.y = y
        self.id = -1
        self.reward = 0
        self.wins = 0
        self.n_class = n_class  # set to zero for regression
        self.dataset_counts = torch.zeros(n_class)

    def add_data(self, x, y):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.n_class > 0:  # set n_class = 0 for regression
            self.dataset_counts[y.long()] += 1

        if self.X is None:
            self.X = x
            self.y = y.float()
        else:
            self.X = torch.cat((self.X, x), dim=0)
            self.y = torch.cat((self.y, y.float()))

    def _update(self, x, y):
        pass

    def _seed_train(self):
        pass

    def predict(self, x):
        pass

    def get_reward(self, r):
        self.reward += r


class Net(nn.Module):
    """Simple Network class."""

    def __init__(self, x_dim=28**2, n_class=10, hidden=512,
                 n_layers=2, task='C'):
        super(Net, self).__init__()
        self.x_dim = x_dim
        width = hidden if n_layers > 0 else n_class    
        self.layers = [nn.Linear(x_dim, width)]
        for i in range(n_layers): # num hidden layers
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Linear(width,width))
        if n_layers > 0:
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Linear(width,n_class))
        if task == 'C':
            self.layers.append(nn.Softmax(dim=1))
        elif task == 'R':
            raise ValueError('Regression not yet supported')
        else:
            raise ValueError('Not supported task type')

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        """Returns output after forward pass throug simple network."""
        x = x.view(x.shape[0], -1) #flatten
        return self.mlp(x.float())

class mlpAgent(SLAgent):
    def __init__(self, X=None, y=None, x_dim=28, n_class=10, n_layers=0,
                 hidden=512, task='C', epoch=10, lr=0.001, bsize=256,
                 retrain_limit=25, retrain_max=10**7):
        super(mlpAgent, self).__init__(X=X, y=y, n_class=n_class)

        self.retrain_count = 0
        self.retrain_max = retrain_max
        self.retrain_limit = retrain_limit
        self.task = task
        self.epoch = epoch
        self.x_dim = x_dim
        self.network = Net(n_layers=n_layers, hidden=hidden, x_dim=x_dim, n_class=n_class)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum') if task == 'C' \
            else torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.accuracy = 0
        self.tested = 0
        self.batch_size = bsize

    def predict(self, x, prob=False):
        """Returns the prediction on the inputs x."""
        # TODO: Fix predict with non-mnist data 
        output = self.network(x)
        if prob is True:
            return output
        else:
            _,output = torch.max(output, 1)
            return output

    def _update(self, x, y, is_from_train=False):
        self.network.train()
        self.optimizer.zero_grad()

        # Forward pass
        y = torch.reshape(y.long(),(x.shape[0],))
        self.loss = self.criterion(self.network(x), y)
        self.loss.backward()
        self.optimizer.step()

        if is_from_train is False:
            self.retrain_count += 1
            if self.retrain_count == self.retrain_limit:
                self._seed_train()
                self.retrain_count = 0

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.zeros_(m.bias)

    def _seed_train(self):
        """Initialize training with seed data."""
        self.network.apply(self.weights_init)
        self.loss = 0.0

        dataset = data.TensorDataset(self.X, self.y)
        train_loader = data.DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     drop_last=False)

        i = 0
        for epoch in range(self.epoch):
            for imgs,labels in iter(train_loader):
                self._update(imgs, labels, is_from_train=True)
                i += len(labels)
            y_pred = self.predict(self.X)

            acc = torch.mean((y_pred == self.y.squeeze()).float())
            print(f'epoch: {epoch}; train acc: {acc.item()}')

            if i >= self.retrain_max:
                break


class buying_mlpAgent(mlpAgent):
    def __init__(self, X=None, y=None, x_dim=28, n_class=10, n_layers=0,
                 hidden=512, task='C', epoch=10, lr=0.001, bsize=256,
                 retrain_limit=25, retrain_max=10**7,
                 n_budget=50, AL_strategy='first', AL_parameter=0.75):
        super(bidding_mlpAgent, self).__init__(X=X, y=y, x_dim=x_dim, n_class=n_class,
                                                n_layers=n_layers, hidden=hidden, task=task,
                                                epoch=epoch, lr=lr, bsize=bsize, retrain_limit=retrain_limit,
                                                retrain_max=retrain_max)
        self.initial_budget = n_budget
        self.currnet_budget = n_budget
        self.AL_strategy = AL_strategy
        self.AL_parameter = AL_parameter

    def purchase(self, y_hat=None):
        '''
        determines to purchase or not
        '''
        purchase = 0
        if self.currnet_budget > 0:
            if self.AL_strategy == 'first':
                purchase = 1
            elif self.AL_strategy == 'random':
                if np.random.uniform(size=1)[0] < self.AL_parameter:
                    purchase = 1
            elif self.AL_strategy == 'uncertainty':
                y_hat = y_hat.detach().numpy()
                entropy = -float(np.sum(y_hat*np.log(y_hat+1e-6)))
                if (entropy > -self.AL_parameter*np.log(1/self.n_class)): 
                    purchase = 1
            else:
                assert False, f'Check bidding AL_strategy. The current {self.AL_strategy}'

        return purchase


