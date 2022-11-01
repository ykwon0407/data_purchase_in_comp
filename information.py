import numpy as np
from scipy.special import binom
from scipy.stats import entropy

def xlogx(x):
    rtn = 0
    if x > 0:
        rtn = x * np.log2(x)
    return rtn

def conditional_entropy(alpha, n):
    S = 2*np.log2(n)
    for m in range(1,n):
        #print(m)
        #print(m*np.exp(alpha) +n - m)
        #print(m*np.exp(alpha))
        p = m*np.exp(alpha)/(m*np.exp(alpha) + n - m)
        #print(p)
        H = entropy([p, 1-p], base=2)
        #print(H)
        S += binom(n,m) * (H + (p*np.log2(m) + (1-p)*np.log2(n-m)))
    S /= 2**n
    #print(S)
    return S

def mutual_info(alpha, n):
    return np.log2(n) - conditional_entropy(alpha, n)

def normalized_mutual_info(alpha, n):
    #returns (log(n) - H(A|Y))/log(n) 
   return (mutual_info)/np.log2(n) 
       

def mi2eff(bitrate, n, tol):
    if n == 1:
        rtn = 0 #alpha doesn't matter for n =1 case
    elif n == 2 and bitrate >= 0.5:
        rtn = 32 #max possible bitrate for this case is already 0.5
    else: 
        rtn = binary_line_search(bitrate, lambda x : mutual_info(x,n), tol=tol)
    return rtn
    
def _midpoint(x,y):
    R = max(x,y)
    L = min(x,y)
    return (R-L)/2 + L

def binary_line_search(target, f, x_min=0, x_max=100, tol=1e-3):
    #binary line search assuming f is monotone incr in x 
    x =_midpoint(x_min, x_max)
    val = f(x)
    while np.abs(val - target) > tol:
        if val > target:
            x_max = x
        elif val < target:
            x_min = x
        x = _midpoint(x_min, x_max)
        val = f(x)
    return x
  
