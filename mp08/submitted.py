'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    d = dict()
    tag_d = dict()
    for sentence in train:
        for word, tag in sentence:
            if word not in d:
                d[word] = dict()
            if tag not in d[word]:
                d[word][tag] = 0
            d[word][tag] += 1
    for word in d.keys():
        counter = d[word]
        max_tag, max_count = None, 0
        for tag in counter.keys():
            if counter[tag] > max_count:
                max_count = counter[tag]
                max_tag = tag
            if tag not in tag_d:
                tag_d[tag] = 0
            tag_d[tag] += counter[tag]
        d[word] = max_tag
    max_frequency, most_frequent_tag = 0, None
    for tag in tag_d:
        if tag_d[tag] > max_frequency:
            max_frequency = tag_d[tag]
            most_frequent_tag = tag

    prediction = []
    for sentence in test:
        new_sentence = []
        for word in sentence:
            if word not in d:
                new_sentence.append((word, most_frequent_tag))
            else:
                new_sentence.append((word, d[word]))
        prediction.append(new_sentence)
    return prediction


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



