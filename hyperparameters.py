import numpy as np
from time import time
from deep_network import Network
import itertools 
import random

def random_hyperparameters(n):

    def sample_logspace(min_log, max_log, n):
        r = (min_log - max_log) * np.random.rand(n) + max_log 
        sample = np.power(10, r)   # because 1 - beta = 10^r
        return sample

    learning_rates = sample_logspace(-5, -2, n)
    regs = sample_logspace(-2, 1, n)
    # hidden_layers = np.random.randint(20, 60, size=(n, 2))

    return learning_rates, regs

def find_hyperparameters(X_tr, Y_tr, X_val, Y_val, n):
    print("Starting search for optimal hyperparameters")
    with open('best.csv', 'a') as file:
        file.writelines('counter, best train, best val, hidden laters, reg, lr, xav, method, drop\n')

    learning_rates, regs = random_hyperparameters(n)
    hidden_layers = [[100], [100, 50]]
    dropouts = [1, 0.8, 0.5]

    best_train = 1
    best_val = 1
    
    average = 0
    beta = 0.8

    list_of_hyperparameters = [learning_rates, regs, hidden_layers, dropouts, [True, False], [True, False], ['sgd', 'adam']]

    # total_iterations = n**2 * len(hidden_layers) * len(dropouts) * 2**3 #all combinations of hyperparamrters
    counter = 0
        # for reg in regs:
        #     for drop in dropouts:
        #         for lr in learning_rates:
        #             for xav in [True, False]:
        #                 for bn in [True, False]:
        #                     for method in ['SGD', 'Adam']:
    all_combinations = list(itertools.product(*list_of_hyperparameters))
    total_iterations = len(all_combinations)
    random.shuffle(all_combinations)


    for combination in all_combinations:
        lr, reg, h, drop, xav, bn, method = combination
    
        optim_parameters = {'learning_rate': lr,
                            'reg': reg,
                            'lr_decay': 1,
                            'dropout': drop,
                            'method': method,
                            'use_batchnorm': bn
                            }      

        network = Network(h, xavier=xav)

        tick = time()

        network.train(X_tr, Y_tr, X_test=X_val, Y_test=Y_val, optim_parameters=optim_parameters, 
                            verbose=False, epochs=20, batch_size=128)

        toc = time()

        logger = network.get_logger()

        elapsed_time = toc - tick
        average = beta * average + (1 - beta) * elapsed_time
        average_corrected = average / (1 - beta**(counter+1)) 
        estimated_time = average_corrected * (total_iterations - counter)

        print('{:d}/{:d} ET: {:.2f}s hidden: {}  reg: {:.4f}  lr: {:.4f}  xav: {}  met:{}, bn:{}'.format(
            counter, total_iterations, estimated_time, h, reg, lr, xav, method, bn))

        counter += 1

        val_error, train_error = logger.get_errors()
        if val_error < best_val:
            best_val, best_train = val_error, train_error
            logger.plot(name=str(counter))

            h_str = ' '.join(map(str, h))
            with open('best.csv', 'a') as file:
                file.writelines('{},{},{},{},{},{},{},{},{}\n'.format(counter, best_train, best_val, h_str, reg, lr, xav, method, drop))