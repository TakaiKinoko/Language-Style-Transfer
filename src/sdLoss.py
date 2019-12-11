import random
import numpy as np

'''
getTargetStyle initialize the target style to seed random array of set dimension (500, in the paper)
sdLoss computes the style discrepancy loss of the decomposed style representation Ys for x in Xs
'''

def getTargetStyle(dim = 500, seed = 10):
    '''
    :param dim: dimension of the style representation (y*)
    '''
    random.seed(seed)
    y_target = []

    for _ in range(dim):
        y_target.append(random.random())

    return y_target

def sdLoss(p_ds, y_target, Ys):
    '''
    computes the style discrepancy loss
    :param p_ds: an array of probabilities, output from discriminator D_s, denoting the possibility that a source sentence has the target style
    :param y_target: style representation for the target domain
    :param Ys: an array of style embeddings for each sentence in Xs, output from Ey (style encoder)
    '''
    dis = [] # list of style discrepencies

    for (p, ys) in zip(p_ds, Ys):
        dis.append(p * pow(styleDiscrepancy(ys, y_target), 2))

    return np.mean(dis)

def styleDiscrepancy(ys, y_target):
    '''
    compute the style discrepancy between two style embeddings
    :param ys: output vector of the style encoder, which is the style embedding of source data
    :param y_target: style representation of the target domain
    :return: d(ys, y_target) = l2-norm(ys - y_target)
    '''
    return np.linalg.norm(np.subtract(ys, y_target))
