from abc import ABC, abstractmethod
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class mab_user(ABC):
    def __init__(self, n_arms, lamb=1):
        super(mab_user, self).__init__()
        self.t = torch.tensor(1.0)
        self.r = torch.zeros(n_arms)
        self.n = torch.zeros(n_arms)
        self.id = -1
        self.returns = 0
        self.lamb = lamb

    @abstractmethod
    def choose(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass 

class perfect_user(mab_user):
    # users that always make perfect decision -- can be paired with recEngines 
    # in CF simulations
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def setup_learners(self, learners):
        #this setup routine must be called before perfect_user can run
        self.learners = learners

    def choose(self):
        l_max = [0]*len(self.learners)
        for i,learner in enumerate(self.learners):
            l_max[i] = torch.max(learner.U[self.id] @ learner.V.t())
        return torch.argmax(torch.tensor(l_max))

    def update(self, arm, reward):
        pass 


class ucb_user(mab_user):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def _ranking(self):
        return self.r + self.lamb*torch.sqrt(2*torch.log(self.t)/self.n)

    def choose(self):
        return torch.argmax(self._ranking())

    def update(self, arm, reward):
        self.r[arm] = self.r[arm]*(self.n[arm]) + reward
        self.n[arm] += 1
        self.r[arm] /= self.n[arm]
        self.t += 1
        self.returns += reward

class e_greedy_user(ucb_user):
    def __init__(self, n_arms, eps_scaling=0.333, r_tol=1e-20, eps0=1.0):
        super().__init__(n_arms)
        self.eps_scaling = eps_scaling
        self.eps = eps0
        self.eps0 = eps0
        self.n_arms = n_arms
        self.r_tol = r_tol

    def choose(self):
        if random.random() > self.eps:
            a = torch.argmax(self.r + self.r_tol*torch.randn(self.r.shape))
        else:
            a = random.randint(0,self.n_arms-1)
        return a

    def update(self, arm, reward):
        super().update(arm, reward)
        self.eps = self.eps0/(self.t**self.eps_scaling)

class sw_ucb_user(mab_user):

    def __init__(self, n_arms):
        super(ucb_user, self).__init__()
        self.n_arms = n_arms
        self.t = torch.tensor(1.0)
        self.tau
        self.sw_r = []
        self.sw_arms = []
        self.n = torch.zeros(self.n_arms)
        self.r = torch.zeros(self.n_arms)
        self.alpha = 0.9
        self.lamb = 1
        self.id = -1
        self.returns = 0

    def _ranking(self):
        return self.r/self.n + self.lamb*torch.sqrt(
                (1+self.alpha)*torch.log(self.t)/self.n)

    def update(self, arm, reward):
        self.sw_arm.append(arm)
        self.sw_r.append(reward)
        self.r[arm] += reward
        self.returns += reward
        self.n[arm] += 1
        tau_prime = torch.min(torch.ceil(self.lamb*(self.t**self.alpha)),self.t)
        delta_tau = tau_prime - self.tau
        if delta_tau < 1.0:
            arm = self.sw_arm.pop(0)
            self.r[arm] -= [self.sw_r.pop(0)]
            self.n[arm] -= 1
        self.tau = tau_prime
