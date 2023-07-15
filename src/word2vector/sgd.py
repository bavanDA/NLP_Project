#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
import os.path as op
import numpy as np
import random
import glob
import pickle
from src.log.log import *

SAVE_PARAMS_EVERY = 5000


def load_saved_params(category):
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    params_file = f"models/{category}.word2vec.npy" 
    state_file = f"models/{category}.word2vec.pickle" 
    params = np.load(params_file)
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    return params, state



def save_params(iter, params,category):
    params_file = f"models/{category}.word2vec.npy"
    np.save(params_file, params)
    with open( f"{category}.word2vec.pickle" % iter, "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10,category='all'):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    start_iter = 0

    x = x0

    if not postprocessing:
        def postprocessing(x): return x

    exploss = None
    try:
        log(f"class:{category}", "word2vec")
        for iter in range(start_iter + 1, iterations + 1):
            # You might want to print the progress every few iterations.

            loss = None
            # YOUR CODE HERE (~2 lines)
            loss, gradient = f(x)
            x = x - step * gradient
            # END YOUR CODE

            x = postprocessing(x)
            if iter % PRINT_EVERY == 0:
                if not exploss:
                    exploss = loss
                else:
                    exploss = .95 * exploss + .05 * loss
                print("iter %d: %f" % (iter, exploss))
                log("iter %d: %f" % (iter, exploss), "word2vec")

            if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
                save_params(iter, x,category)

            if iter % ANNEAL_EVERY == 0:
                step *= 0.5

        log(f"class:{category} with {iterations} iters. Loss: {exploss}", "word2vec")
        log(f"---------------------------------------------------------","word2vec")
        return x
    except Exception as e:
        log(str(e), "word2vec")
