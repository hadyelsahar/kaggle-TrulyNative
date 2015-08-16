# -*- coding: utf-8 -*-

"""
This python file is for training a classifier for ads/nonads page
classification for the  Kaggle "Truly Native" competition Files doesn't
have complicated technique other than applying all features/classifiers
and posting cross validation error
"""

import argparse
import csv

import matplotlib.pyplot as plt

from nltk.tokenize import TreebankWordTokenizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_curve, auc

from classes.Document import *
from classes.Utils import *
from classes.DeltaTfidf import *


parser = argparse.ArgumentParser(description='script to train classifier for ads/nonads in '
                                             'stumbleupon using different features and classifiers')
parser.add_argument('-i', '--trainfile', help='training file path', required=True)
parser.add_argument('-u', '--unlabeledfile', help='unlabeled file path', required=False)
parser.add_argument('-f', '--datapath', help='all file folder path', required=True)
parser.add_argument('-o', '--output', help='output file name', required=True)
args = parser.parse_args()

# reading output file name and printing first line
fout = open(args.output, "w")
writer = csv.writer(fout)
writer.writerow(["file"])

####################
# SETTING FEATURES #
####################


vectorizer = CountVectorizer(
    tokenizer=TreebankWordTokenizer().tokenize,
    ngram_range=(1, 2),
    preprocessor = Document.preprocess
)

###########################
# SETTING DATASETs Setups #
###########################

x_train, x_valid, y_train, y_valid = create_train_test_dataset(filename=args.trainfile,
                                                               datapath=args.datapath, percentage=0.2)

x_unlabeled = create_unlabeled_dataset(filename=args.unlabeledfile, datapath=args.datapath)

#######################
# SETTING CLASSIFIERS #
#######################

classifier_valid = GridSearchCV(
    LogisticRegression(penalty="l1", dual=False),
    [{'C': [0.1, 1, 1.5, 2, 3]}],
    scoring="roc_auc"
)

classifier = LogisticRegression(penalty="l1", dual=False)


########################################
# Validation and picking of parameters #
########################################
Document.pageleft = len(y_valid)
Document.status = "validation"

pipeline = Pipeline([
    ('features', vectorizer),
    ('classifier', classifier_valid)
])

pipeline.fit(x_valid, y_valid)
best_validated_classifier = pipeline.named_steps['classifier']
C_best = best_validated_classifier.best_params_["C"]
classifier.set_params(C=C_best)
print "%s is the C best value " % C_best

print "finished validation now training time"

#######################
# SETTING CLASSIFIERS #
#######################

pipeline = Pipeline([
    ('features', vectorizer),
    ('classifier', classifier)
])

###############
# SHOWTIME !! #
###############
Document.pageleft = len(y_train)
Document.status = "training"
pipeline.fit(x_train, y_train)

print ("finished training now prediction time")
Document.pageleft = 0
Document.status = "prediction"
pred = pipeline.predict_proba(x_unlabeled)

for i in pred:
    writer.writerow(i)

fout.close()







