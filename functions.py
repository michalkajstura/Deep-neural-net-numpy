import numpy as np


class ReLU:
    @staticmethod
    def apply(z):
        A = np.maximum(0, z)
        return A
    
    @staticmethod
    def derivative(z):
        dZ = np.ones_like(z)
        dZ[z <= 0] = 0
        return dZ

class Softmax:
    
    @staticmethod
    def apply(a, y):

        a_shift = a - np.max(a, axis=1, keepdims=True)
        exp_scores = np.exp(a_shift)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        mask = np.arange(0, a.shape[0])
        correct_logprobs = -np.log(probs[mask, y])
        
        cost = np.sum(correct_logprobs)
        cost /= a.shape[0]

        return cost
  
    @staticmethod
    def derivative(a, y):

        a_shift = a - np.max(a, axis=1, keepdims=True)
        exp_scores = np.exp(a_shift)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        dA = probs.copy()
        mask = np.arange(a.shape[0])
        dA[mask, y] -= 1
        dA /= a.shape[0]

        return dA