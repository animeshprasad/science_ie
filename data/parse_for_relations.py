import numpy
import os
import io
import shutil
import nltk
import sys
import copy
import difflib
import random
from nltk.corpus import stopwords
from nltk.stem import *
sys.path.append(os.path.abspath("data/scienceie2017_scripts/"))
#from xml_utils import parseXML

def first_index(s1, s2):
    stemmer=PorterStemmer()
    try:
        s1=unicode(s1)
        s2=unicode(s2)
    except UnicodeDecodeError:
        return '0'
    s_1 = set([stemmer.stem(plural) for plural in s1.split('_')])
    s_2 = set([stemmer.stem(plural) for plural in s2.split('_')])
    #print s_1 & s_2 -set(stopwords.words('english'))
    if s_1 & s_2 - set(stopwords.words('english')):
       return str(len(s_1 & s_2 -set(stopwords.words('english'))))
    else:
        return '0'

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return str('{0:.2f}'.format(size*1.0/min(len(s1), len(s2))))+ ' ' +  str('{0:.2f}'.format(size*1.0/max(len(s1), len(s2))))


def SFLF1(s1, s2):
    chars=[]
    if s1.upper() == s1:
        ff=s1
        nf=s2
    elif s2.upper() == s2:
        ff=s2
        nf=s1
    else:
        return 0
    for items in nf:
        if items.upper() == items and items.isalnum():
            chars.append(items)
    if ff == ''.join(chars):
        return 1
    else:
        return 0

def SFLF2(s1, s2):
        if len(s1) > 5 and len(s2) > 5:
            return 0
        if len(s1) > len(s2):
            s1,s2=s2,s1
        i=0
        j=0
        while i<len(s1) and j<len(s2):
            if s1[i].lower() == s2[j].lower():
                i=i+1
                j=j+1
            else:
                j=j+1
        if i == len(s1):
            return 1
        else:
            return 0



def create_feature_list(text):
    k=text.split(" ")
    if k[0] == 'Synonym-of':
        k[0] = '1'
    elif k[0] == 'Hyponym-of':
        k[0] = '2'
    else:
        k[0] = '0'



    if SFLF1(k[1], k[2]) or SFLF2(k[1], k[2]):
        return get_overlap(k[1], k[2]) + " " + '1' + " " + first_index(k[1], k[2]) + " " +k[0] + '\n'
    else:
        return get_overlap(k[1], k[2]) + " " + '0' + " " + first_index(k[1], k[2]) + " " + k[0] + '\n'


def create_feature_test(s1,s2):
    if SFLF1(s1, s2) or SFLF2(s1, s2):
        return get_overlap(s1, s2) + " " + '1' + " " + first_index(s1, s2)  + '\n'
    else:
        return get_overlap(s1, s2) + " " + '0' + " " + first_index(s1, s2)  + '\n'



def parseAnnTest(textfolder = "data/scienceie2017_test/test/"):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''

    flist = os.listdir(textfolder)
    for f in flist:
        if f[-1] != 't':
            continue
        f_text = open(os.path.join(textfolder, f), "rU")

        # there's only one line, as each . ann file is one text paragraph
        for l in f_text:
            text = unicode(l,"utf-8")

        label = numpy.zeros(len(text),dtype="int")
        #TODO: Remove dummy labels


        f_out = io.open(os.path.join(textfolder, "../feat/" + f.replace(".txt", ".out")), "w", encoding='utf-8')
        length=0
        #print label
        #print text
        #TODO: write own tokeniser instead of punkt
        t_text = nltk.word_tokenize(text)
        for lines in t_text:
            while text[length:length+len(lines)] != lines:
                length = length + 1
            f_out.write(unicode(lines)+"\t")
            if len(numpy.unique(label[length:length+len(lines)]))!=1:
                #print lines, label[length:length+len(lines)]
                #TODO: Either change nltk tokeniser to 'something "something as separte token or do it manually here
                fallout = numpy.max(label[length:length+len(lines)])
                f_out.write(unicode(str(fallout), "utf-8") + "\n")
                continue
            f_out.write(unicode(str(int(label[length])),"utf-8")+"\n")
            length = length + len(lines)



#Train test separte line


def parseAnn(textfolder = "data/scienceie2017_train/train/"):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''
    #outf = textfolder.replace('/dev/', '/rel/')
    outf = textfolder.replace('/train2/','/rel/')
    f_out = open(outf + 'feats', "w")

    flist = os.listdir(textfolder)
    for f in flist:
        if f[-1] != 'n':
            continue
        f_anno = open(os.path.join(textfolder, f), "rU")
        f_text = open(os.path.join(textfolder, f.replace(".ann", ".txt")), "rU")




        # there's only one line, as each . ann file is one text paragraph
        for l in f_text:
            text = unicode(l,"utf-8")

        label = numpy.zeros(len(text),dtype="int")
        label_to_class = {
            'Process':1,
            'Task':2,
            'Material':3,
        }

        res_full_anno = []
        res_anno = []
        spans_anno = []
        rels_anno = []
        remain=[]
        seperate = []


        for l in f_anno:
            r_g = l.strip().split("\t")
            r_g_offs = r_g[1].split(" ")

            res_full_anno.append(l.strip())
            # normalise relation instances by looking up entity spans for relation IDs
            if r_g_offs[0].endswith("-of"):
                arg1 = r_g_offs[1].replace("Arg1:", "")
                arg2 = r_g_offs[2].replace("Arg2:", "")
                for l in res_full_anno:
                    r_g_tmp = l.strip().split("\t")
                    if r_g_tmp[0] == arg1:
                        ent1 = r_g_tmp[2].replace(" ", "_")
                    if r_g_tmp[0] == arg2:
                        ent2 = r_g_tmp[2].replace(" ", "_")


                spans_anno.append(" ".join([ent1, ent2]))
                res_anno.append(" ".join([r_g_offs[0], ent1, ent2]))
                rels_anno.append(" ".join([r_g_offs[0], ent1, ent2]))

            else:
                a=l.strip().split("\t")[2]
                remain.append(a.replace(" ","_"))

        for item in remain:
            for tuples in rels_anno:
                if item not in tuples:
                    if random.random() < 0.1:
                        t=tuples.split(" ")
                        if item == t[1] or item == t[2]:
                            pass
                        seperate.append(" ".join(['No', t[1], item]))
                        seperate.append(" ".join(['No', item, t[2]]))


        gg= rels_anno + seperate
        for t_features in gg:
            f_out.write(create_feature_list(t_features))





if __name__ == '__main__':

    if not os.path.exists('../data/scienceie2017_dev/rel/'):
        os.makedirs('../data/scienceie2017_dev/rel/')
    if not os.path.exists('../data/scienceie2017_train/rel/'):
        os.makedirs('../data/scienceie2017_train/rel/')


    #parseAnn('../data/scienceie2017_dev/dev/')
    parseAnn('../data/scienceie2017_train/train2/')


    if not os.path.exists('../data/scienceie2017_test_unlabelled/rel/'):
        os.makedirs('../data/scienceie2017_test_unlabelled/rel/')


    parseAnnTest('../data/scienceie2017_test_unlabelled/test/')
