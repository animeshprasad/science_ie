import numpy
import os
import io
import nltk
from lxml import etree
import nltk

class tfidf:
  def __init__(self):
    self.weighted = False
    self.documents = []
    self.corpus_dict = {}

  def addDocument(self, doc_name, list_of_words):
    # building a dictionary
    doc_dict = {}
    for w in list_of_words:
      doc_dict[w] = doc_dict.get(w, 0.) + 1.0
      self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

    # normalizing the dictionary
    length = float(len(list_of_words))
    for k in doc_dict:
      doc_dict[k] = doc_dict[k] / length

    # add the normalized document to the corpus
    self.documents.append([doc_name, doc_dict])

  def similarities(self, list_of_words):
    """Returns a list of all the [docname, similarity_score] pairs relative to a list of words."""

    # building the query dictionary
    query_dict = {}
    for w in list_of_words:
      query_dict[w] = query_dict.get(w, 0.0) + 1.0

    # normalizing the query
    length = float(len(list_of_words))
    for k in query_dict:
      query_dict[k] = query_dict[k] / length

    # computing the list of similarities
    sims = []
    for doc in self.documents:
      score = 0.0
      doc_dict = doc[1]
      for k in query_dict:
        if k in doc_dict:
          score += (query_dict[k] / self.corpus_dict[k]) + (doc_dict[k] / self.corpus_dict[k])
      sims.append([doc[0], score])

    return sims



def parseXml(textfolder = "data/scienceie2017_train/train/"):
    '''
    Read .xml files
    :param textfolder:
    :return:
    '''

    flist = os.listdir(textfolder)
    list_to_return=[]
    for f in flist:
        if f[-3:] != 'xml':
            continue
        #f_xml = open(os.path.join(textfolder, f), "rU")

        # there's only one line, as each . ann file is one text paragraph
        #for l in f_xml:
        #    text = unicode(l,"utf-8")

        tree = etree.parse(os.path.join(textfolder, f))
        notags = etree.tostring(tree, encoding='utf8', method='text')
        #root = tree.getroot()
        #print (root.findall('xocs:normalized-srctitle'))
        #print root.tag
        #for childs in root:
        #    print childs.tag
        #for items in root.findall('{http://www.elsevier.com/xml/xocs/dtd}normalized-srctitle'):
        #    print items.text
        #print(notags)
        for tokens in nltk.word_tokenize(unicode(notags,"utf-8")):
            list_to_return.append(tokens)
        print 'done'

    return list_to_return


if __name__ == '__main__':
    table = tfidf()
    a=parseXml('data/scienceie2017_dev/dev/')
    b=a
    #b=parseXml('data/scienceie2017_train/train2/')
    for i,items in enumerate(a+b):
        table.addDocument(str(i),items)

    print table.similarities(['Co', 'Ni'])
