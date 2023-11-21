import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.01*x, x)

def ELU(x):
    return (x>0)*x + (x<=0)*(0.01*(np.exp(x)-1))

def swish(x):
    return x*sigmoid(x)

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    return exp_x/sum_exp_x