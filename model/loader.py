import os
import re
import codecs
import unicodedata
from utils import create_dico, create_mapping, normalise
from features import pos_feature, gaz_binary


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert (len(word) == 2)
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences



def word_mapping(sentences, lower, zeros):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[normalise(x[0], lower, zeros) for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word



def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower, zeros):
    """
    Prepare a sentence for evaluation.
    """
    words = [word_to_id[normalise(w, lower, zeros) if normalise(w, lower, zeros) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    pos_tags = pos_feature(str_words)
    gaz = [gaz_binary(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'pos_tags': pos_tags,
        'gaz': gaz,
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower, zeros):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[normalise(w, lower, zeros) if normalise(w, lower, zeros) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        pos_tags = pos_feature(str_words)
        gaz = [gaz_binary(w) for w in str_words]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
            'pos_tags': pos_tags,
            'gaz': gaz,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words, lower, zeros):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)
    
	
	# Load pretrained embeddings from file
    count = 0
    pretrained = set()
    for line in open(ext_emb_path, 'r'):
        line = line.decode('utf8', 'ignore')
        try:
            #Embeeding size should be greater than 20
#            if line.rstrip()=='' or len(line.rstrip().split()) < 20:
#                print line
#            else:
                pretrained.add(line.rstrip().split()[0].strip())
                count = count + 1
        except:
            pass
    print str(count) + ' Words loaded'



    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            word = normalise(word,lower,zeros)
            if word in pretrained and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    #Dictionary will only have word which are already there (from training words load)
    #or inside the embedding file for test or dev tokens, anything not in train dev and test
    #would be taken care of by UNK inserted during training words load
    return dictionary, word_to_id, id_to_word