import numpy as np

def logistic(x, d=0):
    """
    Input: x is a numpy array. \n
    Input: if d != 0, return derivative of function. \n
    Logistic:  f(x) = 1/(1+e^(-x)) \n
    Derivative: f'(x) = f(x)*(1-f(x)) \n
    """
    x *= -1
    vmax = np.max(x)
    x -= vmax
    fx = 1/(1+np.exp(x))
    if d == 0:
        return fx
    else:
        return fx*(1-fx)

def softmax(x):
    """Softmax function
    Input: x is a numpy 1-D array \n
    Softmax: f(x) = e^(x) / sum(e^x)
    """
    vmax = np.max(x, axis=0)
    x -= vmax
    x_exp = np.exp(x)
    fx = x_exp / np.sum(x_exp, axis=0)
    return fx
    
def tanh(x, d = 0):
    """
    Input: x is a numpy array. \n
    Input: if d != 0, return derivative of function. \n
    TanH:  f(x) = tanh(x) = 2/(1+e^(-2x))-1 \n
    Derivative: f'(x) = 1 - [f(x)]^2 \n
    """
    fx = np.tanh(x)
    if d == 0:
        return fx
    else:
        return 1-np.power(fx, 2)