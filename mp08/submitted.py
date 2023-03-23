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
    # Laplace smoothing coefficient
    k = 1e-5

    # initial state probability = 1 because every sentence starts with 'START'
    pi = 0.99

    # count tag pairs, a[tag1][tag2] = number of times tag2 is after tag1
    a = dict()
    for sentence in train:
        for i in range(len(sentence)-1):
            tag1 = sentence[i][1]
            tag2 = sentence[i+1][1]
            if tag1 not in a:
                a[tag1] = dict()
            if tag2 not in a[tag1]:
                a[tag1][tag2] = 0
            a[tag1][tag2] += 1
    # normalize a under Laplace smoothing, a[tag1][tag2] = probability of tag2 given tag1
    for tag1 in a:
        totalNumOfTags = sum(a[tag1].values()) + k*len(a[tag1].values()) + k
        for tag2 in a[tag1]:
            a[tag1][tag2] = (k+a[tag1][tag2]) / totalNumOfTags
        a[tag1]["OOV"] = k / totalNumOfTags

    # count tag-word pairs, b[tag][word] = number of times tag yield word
    b = dict()
    for sentence in train:
        for word, tag in sentence:
            if tag not in b:
                b[tag] = dict()
            if word not in b[tag]:
                b[tag][word] = 0
            b[tag][word] += 1
    # normalize b number Laplace smoothing, b[tag][word] = probability tag yields word
    for tag in b:
        totalNumOfWords = sum(b[tag].values()) + k*len(b[tag].values()) + k
        for word in b[tag]:
            b[tag][word] = (k+b[tag][word]) / totalNumOfWords
        b[tag]["OOV"] = k / totalNumOfWords

    for tag in b:
        for word in b[tag]:
            if word == "OOV":
                print(f"b[{tag}]['OOV'] = {b[tag]['OOV']}")

    # Viterbi algorithm
    prediction = []
    for sentence in test:
        parent = [dict() for _ in range(len(sentence))]
        v = [dict() for _ in range(len(sentence))]
        # initialization
        for tag in b:
            if tag == 'START':
                v[0][tag] = log(pi)
            else:
                v[0][tag] = log((1-pi)/len(b.keys()))
        # iteration
        for i in range(1, len(sentence)):
            word = sentence[i]
            for tag in b:
                max_prob, max_tag = -float("inf"), None
                for prev_tag in a:
                    if tag not in a[prev_tag]:
                        if word not in b[tag]:
                            x = v[i - 1][prev_tag] + log(a[prev_tag]["OOV"]) + log(b[tag]["OOV"])
                        else:
                            x = v[i - 1][prev_tag] + log(a[prev_tag]["OOV"]) + log(b[tag][word])
                    else:
                        if word not in b[tag]:
                            x = v[i - 1][prev_tag] + log(a[prev_tag][tag]) + log(b[tag]["OOV"])
                        else:
                            x = v[i-1][prev_tag] + log(a[prev_tag][tag]) + log(b[tag][word])
                    if x > max_prob:
                        max_prob = x
                        max_tag = prev_tag
                v[i][tag] = max_prob
                parent[i][tag] = max_tag
        # termination
        last_tag = "END"
        tags = [last_tag]
        for i in range(len(sentence)-2, -1, -1):
            last_tag = parent[i+1][last_tag]
            tags.append(last_tag)
        tags.reverse()
        prediction.append(list(zip(sentence, tags)))

    return prediction


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # Laplace smoothing coefficient
    k = 1e-5

    # Laplace smoothing coefficient for hapax
    k_hapax = 1e-5

    # initial state probability = 1 because every sentence starts with 'START'
    pi = 0.99

    # count tag pairs, a[tag1][tag2] = number of times tag2 is after tag1
    a = dict()
    for sentence in train:
        for i in range(len(sentence) - 1):
            tag1 = sentence[i][1]
            tag2 = sentence[i + 1][1]
            if tag1 not in a:
                a[tag1] = dict()
            if tag2 not in a[tag1]:
                a[tag1][tag2] = 0
            a[tag1][tag2] += 1
    # normalize a under Laplace smoothing, a[tag1][tag2] = probability of tag2 given tag1
    for tag1 in a:
        totalNumOfTags = sum(a[tag1].values()) + k * len(a[tag1].values()) + k
        for tag2 in a[tag1]:
            a[tag1][tag2] = (k + a[tag1][tag2]) / totalNumOfTags
        a[tag1]["OOV"] = k / totalNumOfTags

    # count tag-word pairs, b[tag][word] = number of times tag yield word
    b = dict()
    for sentence in train:
        for word, tag in sentence:
            if tag not in b:
                b[tag] = dict()
            if word not in b[tag]:
                b[tag][word] = 0
            b[tag][word] += 1
    # calculate hapax[T] = P(T|hapax), probability that a hapax word has tag T
    hapax = dict()
    hapax_count = k_hapax
    for tag in b:
        for word in b[tag]:
            if b[tag][word] == 1:
                hapax_count += (1+k_hapax)
    for tag in b:
        hapax_count_for_T = k_hapax
        for word in b[tag]:
            if b[tag][word] == 1:
                hapax_count_for_T += (1+k_hapax)
        hapax[tag] = hapax_count_for_T / hapax_count
    # normalize b number Laplace smoothing, b[tag][word] = probability tag yields word
    for tag in b:
        k_hat = k * hapax[tag]
        totalNumOfWords = sum(b[tag].values()) + k_hat * len(b[tag].values()) + k_hat
        for word in b[tag]:
            b[tag][word] = (k_hat + b[tag][word]) / totalNumOfWords
        b[tag]["OOV"] = k_hat / totalNumOfWords

    # Viterbi algorithm
    prediction = []
    for sentence in test:
        parent = [dict() for _ in range(len(sentence))]
        v = [dict() for _ in range(len(sentence))]
        # initialization
        for tag in b:
            if tag == 'START':
                v[0][tag] = log(pi)
            else:
                v[0][tag] = log((1 - pi) / len(b.keys()))
        # iteration
        for i in range(1, len(sentence)):
            word = sentence[i]
            for tag in b:
                max_prob, max_tag = -float("inf"), None
                for prev_tag in a:
                    if tag not in a[prev_tag]:
                        if word not in b[tag]:
                            x = v[i - 1][prev_tag] + log(a[prev_tag]["OOV"]) + log(b[tag]["OOV"])
                        else:
                            x = v[i - 1][prev_tag] + log(a[prev_tag]["OOV"]) + log(b[tag][word])
                    else:
                        if word not in b[tag]:
                            x = v[i - 1][prev_tag] + log(a[prev_tag][tag]) + log(b[tag]["OOV"])
                        else:
                            x = v[i - 1][prev_tag] + log(a[prev_tag][tag]) + log(b[tag][word])
                    if x > max_prob:
                        max_prob = x
                        max_tag = prev_tag
                v[i][tag] = max_prob
                parent[i][tag] = max_tag
        # termination
        last_tag = "END"
        tags = [last_tag]
        for i in range(len(sentence) - 2, -1, -1):
            last_tag = parent[i + 1][last_tag]
            tags.append(last_tag)
        tags.reverse()
        prediction.append(list(zip(sentence, tags)))

    return prediction



