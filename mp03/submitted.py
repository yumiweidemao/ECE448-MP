'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np


def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - a list of k images, the k nearest neighbors of image
    labels - a list of k labels corresponding to the k images
    '''

    indices = list(range(len(train_images)))
    distances = []
    for train_img in train_images:
        distances.append(np.linalg.norm(image - train_img))
    indices = sorted(indices, key=lambda a: distances[a])
    return train_images[indices[:k]], train_labels[indices[:k]]


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -one majority-vote scores for each of the M dev images
    '''

    hypotheses, scores = [], []
    for image in dev_images:
        _, labels = k_nearest_neighbors(image, train_images, train_labels, k)
        score = len(labels[labels==True])
        if score > k/2:
            hypotheses.append(1)
        else:
            hypotheses.append(0)
        if score == 0:
            score = 2
        scores.append(score)
    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    matrix = np.array([[0 for _ in range(2)] for _ in range(2)])
    for m in range(len(references)):
        matrix[int(references[m])][int(hypotheses[m])] += 1

    accuracy = (matrix[0][0] + matrix[1][1])/ np.sum(matrix)
    precision = matrix[1][1] / (matrix[0][1] + matrix[1][1])
    recall = matrix[1][1] / (matrix[1][1] + matrix[1][0])
    f1 = 2 / (1/recall + 1/precision)
    return matrix, accuracy, f1

