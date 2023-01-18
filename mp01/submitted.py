'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    d = dict()  # key: word count tuple, value: number of texts
    for text in texts:
        word0_cnt = text.count(word0)
        word1_cnt = text.count(word1)
        d[(word0_cnt, word1_cnt)] = d.get((word0_cnt, word1_cnt), 0) + 1

    m = max(d.keys(), key=lambda a: a[0])[0]
    n = max(d.keys(), key=lambda a: a[1])[1]
    Pjoint = np.array([[0.0 for _ in range(n + 1)] for _ in range(m + 1)])
    for k0, k1 in d:
        Pjoint[k0][k1] = d[(k0, k1)] / len(texts)
    return Pjoint


def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    l = len(Pjoint) if index == 0 else len(Pjoint[0])
    Pmarginal = np.array([0.0 for _ in range(l)])
    for x in range(l):
        if index == 0:
            for y in range(len(Pjoint[0])):
                Pmarginal[x] += Pjoint[x][y]
        else:
            for y in range(len(Pjoint)):
                Pmarginal[x] += Pjoint[y][x]

    return Pmarginal


def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pcond = np.array([[0.0 for _ in range(len(Pjoint[0]))] for _ in range(len(Pjoint))])
    for m in range(len(Pjoint)):
        for n in range(len(Pjoint[0])):
            Pcond[m][n] = Pjoint[m][n] / Pmarginal[m]
    return Pcond


def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    mu = 0.0
    for n in range(len(P)):
        mu += n * P[n]
    return mu


def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    mu_x_squared = 0.0
    for n in range(len(P)):
        mu_x_squared += (n**2 * P[n])
    var = mu_x_squared - mean_from_distribution(P)**2
    return var


def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    Pmarginal_0 = marginal_distribution_of_word_counts(P, 0)
    Pmarginal_1 = marginal_distribution_of_word_counts(P, 1)
    Ex0, Ex1 = 0.0, 0.0
    for n in range(len(Pmarginal_0)):
        Ex0 += n * Pmarginal_0[n]
    for n in range(len(Pmarginal_1)):
        Ex1 += n * Pmarginal_1[n]
    covar = 0.0
    for m in range(len(P)):
        for n in range(len(P[0])):
            covar += P[m][n] * (m - Ex0) * (n - Ex1)
    return covar


def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0.0
    for x0 in range(len(P)):
        for x1 in range(len(P[0])):
            expected += P[x0][x1] * f(x0, x1)
    return expected
