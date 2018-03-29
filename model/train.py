#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader

from loader import word_mapping, char_mapping, tag_mapping
from loader import prepare_dataset
from features import write_crfpp_feat_file

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="1",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-i", "--use_pos", default="0",
    type='int', help="Use POS features (0 to disable)"
)

opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['cap_dim'] = opts.cap_dim
parameters['use_pos'] = opts.use_pos

# Check parameters validity
print opts.train
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)


# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
pos = parameters['use_pos']

# Load sentences
train_sentences = loader.load_sentences(opts.train)
dev_sentences = loader.load_sentences(opts.dev)
test_sentences = loader.load_sentences(opts.test)


dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower, zeros)
dico_words_train = dico_words

# Create a dictionary and a mapping for words / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower, zeros
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower, zeros
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower, zeros
)


##Write to CRFPP Feature File
write_crfpp_feat_file(train_data,'atrain')
write_crfpp_feat_file(dev_data,'adev')
write_crfpp_feat_file(test_data,'atest')

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

