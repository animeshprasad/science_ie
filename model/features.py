#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk

def pos_feature(tokens):
    """
    POS tag feature:
    Refer to nltk.pos_tags()
    Using 'averaged_perceptron_tagger'
    """
    pos_dict={
    '$':1,
    'CC':2,
    'CD':3,
    'DT':4,
    'EX':5,
    'FW':6,
    'IN':7,
    'JJ':8,
    'JJR':9,
    'JJS':10,
    'LS':11,
    'MD':12,
    'NN':13,
    'NNS':14,
    'NNP':15,
    'NNPS':16,
    'PDT':17,
    'POS':18,
    'PRP':19,
    'PRP$':20,
    'RB':21,
    'RBR':22,
    'RBS':23,
    'RP':24,
    'SYM':25,
    'TO':26,
    'UH':27,
    'VB':28,
    'VBD':29,
    'VBG':30,
    'VBN':31,
    'VBP':32,
    'VBZ':33,
    'WDT':34,
    'WP':35,
    'WP$':36,
    'WRB':37,
    '``':38,
    '\'\'':39,
    '(':40,
    ')':41,
    ',':42,
    '--':43,
    '.':44,
    ':':45,
    }
    count = 0
    return [pos_dict[a] for (w, a) in nltk.pos_tag(tokens)]


def binary_to_int(value):
    if value:
        return 1
    else:
        return 0

def contains_hyphen(tokens):
    symbols=[u'-', u'–']
    return  binary_to_int(any(symbols[i] in tokens for i in xrange(len(symbols))))

def in_quotes(tokens):
    symbols = [u'\'', u'‘', u'’', u'"', u'`', u'“', u'”']
    return binary_to_int(any(symbols[i] in [tokens[0], tokens[-1]] for i in xrange(len(symbols))))
#    for i,items in enumerate(k):
#        if any (k[i-1] ==1, k[i-2] ==1, k[i-3] ==1) and any (k[i+1] == 1, k[i+2] ==1, k[i+3] ==1):
#            k[i] = 1

def contains_maths(tokens):
    symbols=[u'=', u'>', u'<']
    return binary_to_int(any(symbols[i] in tokens for i in xrange(len(symbols))))

def non_ascii(tokens):
    return binary_to_int(all([False if ord(x) < 128 else True for x in tokens]))


#def domain(tokens):
#    return [pos_dict[a] for (w, a) in nltk.pos_tag(tokens)]


def gaz_binary(tokens):
    return [contains_hyphen(tokens),in_quotes(tokens),contains_maths(tokens),non_ascii(tokens)]

