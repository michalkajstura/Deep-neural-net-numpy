import numpy as np 

def check_grad(X, Y, parameters, forward_cost, eps=1e-6):
    grad_approx = {}
    for key in parameters.keys():

        param = parameters[key]

        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            ix = it.multi_index

            oldval = param[ix]
            param[ix] = oldval + eps
            plus = forward_cost(X, Y, parameters)
            param[ix] = oldval - eps
            minus = forward_cost(X, Y, parameters)

            grad[ix] = (plus - minus) / (2 * eps)
            param[ix] = oldval
            it.iternext()
        
        grad_approx[key] = grad

    return grad_approx

def grad_diff(grad, grad_approx):
    
    for key in grad_approx.keys():
        numerator = np.linalg.norm(grad[key] - grad_approx[key])
        denominator = np.linalg.norm(grad[key]) + np.linalg.norm(grad_approx[key]) 
        diff = numerator / denominator
        print(key + ': ' + str(diff))