import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class Logger:

    def __init__(self):
        # Keep track of losses and errors
        self.loss_history = []
        self.val_error_history = []
        self.train_error_history = []

    def add_loss(self, loss):
        self.loss_history.append(loss)
    
    def add_errors(self, val_error, train_error):
        self.val_error_history.append(val_error)
        self.train_error_history.append(train_error)
    
    def plot(self, name=None):
        matplotlib.rcParams.update({'font.size': 10})
        
        # reject outliers
        mean = np.mean(self.loss_history)
        std = np.std(self.loss_history)
        loss_hist = [x for x in self.loss_history if (x < mean + 2 * std)]

        plt.subplot(2, 1, 1)
        plt.title("Loss history")
        matplotlib.pyplot.grid(b=True, linestyle='dotted')
        plt.plot(loss_hist)

        plt.subplot(2, 1, 2)
        plt.title("Test and train error")
        plt.xticks(np.arange(0, len(self.val_error_history) + 1, 5))
        plt.yticks(np.arange(0, max(self.val_error_history) +1, 5 ))

        train_patch = mpatches.Patch(color='green', label='Train error: ' + str(self.train_error_history[-1])[:6])
        test_patch = mpatches.Patch(color='red', label='Validation error: ' + str(self.val_error_history[-1])[:6])
        plt.legend(handles=[train_patch, test_patch])
        plt.plot(self.train_error_history, c='green')
        plt.plot(self.val_error_history, c='red')
        if name is not None:
            plt.savefig('images/' + name + '.png', dpi=300)
            # plt.savefig('test.pdf', format='pdf')
            plt.close()
        else:
            plt.show()

    def get_errors(self):
        return self.val_error_history[-1], self.train_error_history[-1]