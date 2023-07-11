

from src.config import *
import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from src.word2vector.word2vec import *
from src.word2vector.sgd import *
from src.word2vector.treebank import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
for category in CATEGORIES:
    dataset = SteamData(f'{SENTENCE_BROKEN_PATH}/{category}/{category}.csv')
    tokens = dataset.tokens()
    nWords = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10
    # Context size
    C = 5

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    startTime = time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) /
        dimVectors, np.zeros((nWords, dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                        negSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10,category=category)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("training took %d seconds" % (time.time() - startTime))
    log("training took %d seconds" % (time.time() - startTime),"word2vec")
    log("----------------------------------------------------------------------","word2vec")

