import numpy
import os
import io
import shutil
import nltk
import sys

sys.path.append(os.path.abspath("data/scienceie2017_scripts/"))
from xml_utils import parseXML

def joinFiles(textfolder = "data/scienceie2017_train/feat/"):
    flist = os.listdir(textfolder)
    with open(textfolder+"feats", 'wb') as outfile:
        count  = 0
        for filename in flist:
            if filename[-4:] != '.out':
                continue
            with open(textfolder+filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
                outfile.write("\n")
            readfile.close()
            count = count + 1
    print str(count) + " files added to features"


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

    for f in flist:
        if f[-1] != 'l':
            continue
        f_xml1 = parseXML(textfolder + f, True, False, False)
        f_xml2 = parseXML(textfolder + f, False, True, False)
        f_xml3 = parseXML(textfolder + f, False, False, True)
        f_out = io.open(os.path.join(textfolder, "../temp/" + f.replace(".xml", ".out")), "w", encoding='utf-8')
        f_out.write(unicode(str(f_xml1), "utf-8") + "\n")
        f_out.write(unicode(str(f_xml2), "utf-8") + "\n")
        f_out.write(unicode(str(f_xml3), "utf-8") + "\n")



def parseAnn(textfolder = "data/scienceie2017_train/train/"):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return:
    '''

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

        for l in f_anno:
            anno_inst = unicode(l.strip(),'utf-8').split("\t")
            if len(anno_inst) == 3:
                keytype, start, end = anno_inst[1].split(" ")
                if not keytype.endswith("-of"):
                    # look up span in text and print error message if it doesn't match the .ann span text
                    keyphr_text_lookup = text[int(start):int(end)]
                    keyphr_ann = anno_inst[2]
                    if keyphr_text_lookup != keyphr_ann:
                        print("Spans don't match for anno " + l.strip() + " in file " + f)
                        print keyphr_text_lookup,keyphr_ann
                    label[int(start):int(end)].fill(label_to_class[keytype])


        f_out = io.open(os.path.join(textfolder, "../feat/" + f.replace(".ann", ".out")), "w", encoding='utf-8')
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

    for f in flist:
        if f[-1] != 'l':
            continue
        f_xml1 = parseXML(textfolder+f, True, False, False)
        f_xml2 = parseXML(textfolder + f, False, True, False)
        f_xml3 = parseXML(textfolder + f, False, False, True)
        f_out = io.open(os.path.join(textfolder, "../temp/" + f.replace(".xml", ".out")), "w", encoding='utf-8')
        f_out.write(unicode(str(f_xml1), "utf-8") + "\n")
        f_out.write(unicode(str(f_xml2), "utf-8") + "\n")
        f_out.write(unicode(str(f_xml3), "utf-8") + "\n")


if __name__ == '__main__':
#    if not os.path.exists('data/scienceie2017_dev/feat/'):
#        os.makedirs('data/scienceie2017_dev/feat/')
#    if not os.path.exists('data/scienceie2017_train/feat/'):
#        os.makedirs('data/scienceie2017_train/feat/')
#    if not os.path.exists('data/scienceie2017_dev/temp/'):
#        os.makedirs('data/scienceie2017_dev/temp/')
#    if not os.path.exists('data/scienceie2017_train/temp/'):
#        os.makedirs('data/scienceie2017_train/temp/')

    parseAnn('data/scienceie2017_dev/dev/')
    parseAnn('data/scienceie2017_train/train2/')
    joinFiles('data/scienceie2017_dev/feat/')
    joinFiles('data/scienceie2017_train/feat/')

#    if not os.path.exists('data/scienceie2017_test_unlabelled/feat/'):
#        os.makedirs('data/scienceie2017_test_unlabelled/feat/')
#    if not os.path.exists('data/scienceie2017_test_unlabelled/temp/'):
#        os.makedirs('data/scienceie2017_test_unlabelled/temp/')

    parseAnnTest('data/scienceie2017_test_unlabelled/test/')
    joinFiles('data/scienceie2017_test_unlabelled/feat/')

    joinFiles('data/scienceie2017_train/temp/')
    joinFiles('data/scienceie2017_dev/temp/')
    joinFiles('data/scienceie2017_test_unlabelled/temp/')