# science_ie

This is the code for the CRF based pipeline submitted to SemEval 2017.
The features genenarted from train/test/dev tokens and template are provided.
The feature file should be used to train a CRFPP model using template provided.

Primary Workflow:
python train.py --train ../data/scienceie2017_train/feat/feats --dev ../data/scienceie2017_dev/feat/feats --test ../data/scienceie2017_test_unlabelled/feat/feats



Please cite following publication(s) if you refer this work:
```
@InProceedings{prasad-kan:2017:SemEval,
  author    = {Prasad, Animesh  and  Kan, Min-Yen},
  title     = {WING-NUS at SemEval-2017 Task 10: Keyphrase Extraction and Classification as Joint Sequence Labeling},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {972--976},
  url       = {http://www.aclweb.org/anthology/S17-2170}
}
```
