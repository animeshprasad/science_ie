import numpy
import os
import io
import shutil
import nltk
import sys

sys.path.append(os.path.abspath("data/scienceie2017_scripts/"))
#from xml_utils import parseXML
from parse2 import create_feature_test


def parsetoAnn(outfile='data/scienceie2017_dev/out', ifolder='data/scienceie2017_dev/dev/', ofolder='data/scienceie2017_dev/final/'):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''

    #from sklearn.ensemble import RandomForestClassifier
    #from sklearn.metrics import precision_score, recall_score, f1_score

    #import numpy

    #train_data = numpy.loadtxt('../data/scienceie2017_train/rel/feats')
    #test_data = numpy.loadtxt('../data/scienceie2017_train/rel/feats')

    #print train_data
    #print train_data[:, 0:-1], train_data[:, -1]
    # raw_input()


    # Create the random forest object which will include all the parameters
    # for the fit
    #forest = RandomForestClassifier(n_estimators=30)

    # Fit the training data to the Survived labels and create the decision trees
    #forest = forest.fit(train_data[:, :-1], train_data[:, -1])

    # Take the same decision trees and run it on the test data
    #output = forest.predict(test_data[:, :-1])

    #y_pred = output
    #y_true = test_data[:, -1]
    #print precision_score(y_true, y_pred, average=None)
    #print recall_score(y_true, y_pred, average=None)
    #print f1_score(y_true, y_pred, average=None)


    all=open(outfile,'r')
    content = all.readlines()
    line_no=0
    flist = os.listdir(ifolder)

    for f in flist:
        if f[-1] != 't':
            continue


        f_anno = open(os.path.join(ofolder, f[:-4]+'.ann'), "w")
        f_txt = open(os.path.join(ifolder, f), "r")

        # there's only one line, as each . ann file is one text paragraph
        for lf in f_txt:
            text=unicode(lf,"utf-8")
        label = numpy.zeros(len(text),dtype="int")

        label_to_class = {
         #   1:'KEYPHRASE-NOTYPES'
            1:'Process',
            3:'Task',
            2:'Material',
        }

        start=0
        end=0

        while content[line_no].strip() != '':
            l = content[line_no]
            line_no += 1
            all_labels = unicode(l,"utf-8").strip().split('\t')
            end=start+len(all_labels[0])
            while text[int(start):int(end)] != all_labels[0]:
                #print text[int(start):int(end)], all_labels[0]
                start = start+1
                end = end+1

            keytype = all_labels[-1]
            label[int(start):int(end)].fill(keytype)
            start = end

        line_no+=1

        for i in xrange(len(label)):
            if label[i] == 0 and  text[i]==' ':
                if label[i-1] == label[i+1] and label[i-1]!=0:
                    label[i]= label[i-1]


        start=0
        end=0
        i=0
        all_annotations=[]
        while i <len(label):
            if label[i] != 0:
                start=i
                while label[i+1]!=0:
                    i=i+1
                end=i
                #print str(start)+'\t'+str(start)+' '+str(end)+'\t'+label_to_class[label[start]]+'\t'+text[start:end]
                f_anno.write(str(start)+'\t'+label_to_class[label[start]]+' '+str(start)+' '+str(end+1)+'\n')
                all_annotations.append(str(start)+'\t'+label_to_class[label[start]]+' '+str(start)+' '+str(end+1)+'\n')
                #for items in all_annotations[:-1]:
                #    candidate=items.split('\t')
                #    candidate_name=candidate[0]
                #    candidate_se=candidate[1].split(' ')
                #    second=text[start:end+1]
                #    first=text[int(candidate_se[1]):int(candidate_se[2])+1]
                #    k=numpy.array(create_feature_test(second, first).strip().split(' ') + create_feature_test(first, second).strip().split(' '))
                #    k=k.reshape(2,4)
                #    predcs_rel=forest.predict(k)
                #    if predcs_rel[1] ==2 or predcs_rel[0] ==2:
                #        if candidate_se[0] == 'Material' and label_to_class[label[start]] != 'Material':
                #            pass
                #        f_anno.write( '*\tHyponym-of ' + candidate_name + ' '+str(start)+ '\n')
                #        f_anno.write('*\tHyponym-of ' + str(start) + ' ' + candidate_name + '\n')
                #    elif predcs_rel[1] ==1 or predcs_rel[0] ==1:
                #        if candidate_se[0] == 'Material' and label_to_class[label[start]] != 'Material':
                #            pass
                #        f_anno.write('*\tSynonym-of ' + candidate_name + ' ' + str(start) + '\n')
                #    else:
                #        pass

            i+=1


if __name__ == '__main__':

    if not os.path.exists('../data/scienceie2017_dev/final/'):
        os.makedirs('../data/scienceie2017_dev/final/')
    if not os.path.exists('../data/scienceie2017_test_unlabelled/final/'):
        os.makedirs('../data/scienceie2017_test_unlabelled/final/')


    parsetoAnn('../data/scienceie2017_dev/out3', '../data/scienceie2017_dev/dev/', '../data/scienceie2017_dev/final/')
    #parsetoAnn('../data/scienceie2017_test_unlabelled/out_test', '../data/scienceie2017_test_unlabelled/test/', '../data/scienceie2017_test_unlabelled/final/')