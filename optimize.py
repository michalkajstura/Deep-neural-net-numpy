import matplotlib.pyplot as plt
import numpy as np
from misc import check_grad, grad_diff




def SGD_step(params, grads, optim_params):
    learning_rate = optim_params.get('learning_rate', 1)

    for key in params.keys():
        params[key] -= learning_rate * grads[key]

    return params, optim_params

def Adam_step(params, grads, optim_params):
        s = optim_params.get('s')
        v = optim_params.get('v')
        t = optim_params.get('t', 1)
        optim_params['t'] = t

        learning_rate = optim_params.get('learning_rate', 1)
        beta1 = optim_params.get('beta1', 0.9)
        beta2 = optim_params.get('beta2', 0.99)
        epsilon = optim_params.get('epsilon', 1e-7)

        for p in params.keys():
            v[p] = beta1 * v[p] + (1 - beta1) * grads[p] 
            s[p] = beta2 * s[p] + (1 - beta2) * grads[p]**2

            # correction
            v_hat = v[p] / (1 - beta1**t)
            s_hat = s[p] / (1 - beta2**t)


            params[p] -= learning_rate * (v_hat / np.sqrt(s_hat + epsilon))

            optim_params['s'] = s
            optim_params['v'] = v
            optim_params['t'] += 1

        return params, optim_params