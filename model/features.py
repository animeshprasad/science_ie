#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import re
from nltk.corpus import stopwords


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
    '#':46,
    }
    return [pos_dict[a] for (w, a) in nltk.pos_tag(tokens)]


def binary_to_int(value):
    if value:
        return 1
    else:
        return 0

def contains_hyphen(tokens):
    symbols=[u'-', u'–']
    return  binary_to_int(any(symbols[i] in tokens for i in xrange(len(symbols))))

def in_quotes(tokens, flag_in):
    symbols = [u'\'', u'‘', u'’', u'"', u'`', u'“', u'”']
    if tokens == '\'s':
        return 0
    if any(symbols[i] in [tokens[0]] for i in xrange(len(symbols))) and any(symbols[i] in [tokens[-1]] for i in xrange(len(symbols))):
        return 2
    elif any(symbols[i] in [tokens[0]] for i in xrange(len(symbols))):
        flag_in=1
    elif any(symbols[i] in [tokens[-1]] for i in xrange(len(symbols))):
        flag_in=-1
    else:
        return flag_in
    return flag_in


#def in_quotes(tokens):
#    symbols = [u'\'', u'‘', u'’', u'"', u'`', u'“', u'”']
#    return binary_to_int(any(symbols[i] in [tokens[0], tokens[-1]] for i in xrange(len(symbols))))
#    for i,items in enumerate(k):
#        if any (k[i-1] ==1, k[i-2] ==1, k[i-3] ==1) and any (k[i+1] == 1, k[i+2] ==1, k[i+3] ==1):
#            k[i] = 1



def contains_maths(tokens):
    symbols=[u'=', u'>', u'<']
    return binary_to_int(any(symbols[i] in tokens for i in xrange(len(symbols))))

def non_ascii(tokens):
    return binary_to_int(all([False if ord(x) < 128 else True for x in tokens]))


def alphanumeric(tokens):
    return True if re.match('[a-zA-Z]+$', tokens) and re.match('[0-9]+$', tokens) else False


def gaz_binary(tokens):
    return str(contains_hyphen(tokens))+ ' '  +str(contains_maths(tokens))+ \
           ' ' +str(non_ascii(tokens))+ ' ' +str(alphanumeric(tokens))+ ' ' +str(tokens.isalnum())



def write_crfpp_feat_file(feat_list,filename):
    #TODO: Replacement coerce str to int
    flag_in=0
    rf=open('../data/scienceie2017_test_unlabelled/temp/feats','r')
    all = rf.readlines()
    stop = set(stopwords.words('english'))
    def get_in_ref(index, word):
        title=unicode(all[index], 'utf-8').lower()
        if word in title and word not in stop:
            return '1'
        else:
            return '0'

    def get_in_others(index, word):
        if word.isdigit():
            return '0'
        if word in ['[', ']', '(', ')', ',', '.']:
            return '0'
        ref=unicode(all[index], 'utf-8').lower()
        if ref.strip() == '':
            return '-'
        if word not in stop:
            k=0
            try:
                k=len(re.findall(word, ref))
            except Exception:
                return '-'
            return str(k)
        else:
            return '0'


    w_stream=open(filename,'w')
    for k,item in enumerate(feat_list):
        for i,items in enumerate(item['str_words']):
            binary_feats=gaz_binary(items)
            flag_in=in_quotes(items, flag_in)
            feat_s=items + ' ' + items[0] + ' ' + items[-1] + \
            ' ' + items[:min(2,len(items))] + ' ' + items[max(-2,-len(items)):] + \
            ' ' + items[:min(3, len(items) )] + ' ' + items[max(-3, -len(
                    items)):] +\
                ' ' + items[:min(4, len(items))] + ' ' + items[max(-4, -len(
                    items)):] + \
                   ' ' + str(item['words'][i]) + ' ' + str(item['caps'][i]) + ' ' + str(item['pos_tags'][i]) +\
                   ' ' + binary_feats +' '+str(abs(flag_in))+ ' ' +str(item['tags'][i])
            w_stream.write(feat_s.encode('utf-8'))
            w_stream.write('\n'.encode('utf-8'))
            if flag_in == 2 or flag_in == -1:
                flag_in = 0
        w_stream.write('\n'.encode('utf-8'))
