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
parser.add_argument('-f', '--datapath', help='all file folder path', required=True)
parser.add_argument('-o', '--output', help='output file name', required=True)
args = parser.parse_args()

# reading output file name and printing first line
fout = open(args.output, "w")
writer = csv.writer(fout)
writer.writerow(["preprocessor", "vectorizer", "classifier", "auc"])


####################
# SETTING FEATURES #
####################

vectorizers = {
    "tfidf": TfidfVectorizer(
        tokenizer=TreebankWordTokenizer().tokenize,
        ngram_range=(1, 2), norm="l1",
        preprocessor = Document.preprocess
        ),
    "count": CountVectorizer(
        tokenizer=TreebankWordTokenizer().tokenize,
        ngram_range=(1, 2),
        preprocessor = Document.preprocess
        ),
    "delta-tfidf": DeltaTfidf(
        tokenizer = TreebankWordTokenizer().tokenize,
        preprocessor = Document.preprocess
        )
}

features = {
    # "tfidf": FeatureUnion([
    #     ("tfidf",
    #      vectorizers["tfidf"])]
    #     ),
    # "delta-tfidf": FeatureUnion([
    #     ("delta-tfidf",
    #      vectorizers["delta-tfidf"])]
    #     ),
    "count": FeatureUnion([
        ("count",
         vectorizers["count"])]
        )
    # "tfidf_delta-tfidf_count"FeatureUnion([
    #     ("count",vectorizers["count"]),
    #     ("delta-tfidf",vectorizers["delta-tfidf"]),
    #     ("tfidf",vectorizers["tfidf"]),
    # ])

}


#########################
# SETTING DATASET TYPES #
#########################

dataset_setups = {
    "normal_setup": create_cv_dataset(filename=args.trainfile, datapath=args.datapath, n_folds=5)
}

#######################
# SETTING CLASSIFIERS #
#######################

classifiers = {
    # "svm_cv": GridSearchCV(
    #     LinearSVC(penalty="l1", dual=False),
    #     [{'C': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}]  # range of C coefficients to try
    # ),
    "LREG": GridSearchCV(
        LogisticRegression(penalty="l1", dual=False),
        [{'C': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}]
        # [{'C': [0.1]}]
    )
    # "BernoulliNB": BernoulliNB(alpha=.01),
    # "SGD": SGDClassifier(loss="hinge", penalty="l1"),
    # "KNN": KNeighborsClassifier(n_neighbors=5, algorithm='auto')
}

###############
# SHOWTIME !! #
###############

# Generating different graph shapes for the ROC curve
__c = ['b', 'g', 'r', 'c', 'm', 'y']
__s = ['-', '--', '-.', ',', 'o', 's', 'p']
graph_shapes = [c+s for s in __s for c in __c]

# iterator to mark which combination is it.
i = 0

for fold_name, fold in dataset_setups.items():
    for fvector_name, fvector in features.items():
        for clf_name, clf in classifiers.items():

            pipeline = Pipeline([
                ('features', fvector),
                ('classifier', clf)
            ])

            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            all_tpr = []

            for i, (X_train, y_train, X_test, y_test) in enumerate(fold):

                pipeline.fit(X_train, y_train)
                pred = pipeline.predict_proba(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, pred[:, 1])
                mean_tpr += np.interp(mean_fpr, fpr, tpr)

            mean_tpr /= len(fold)
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1
            mean_auc = auc(mean_fpr, mean_tpr)

            plt.plot(mean_fpr, mean_tpr, graph_shapes[i % len(graph_shapes)],
                     label='%s-%s (area = %0.2f)' % (fvector_name, clf_name, mean_auc), lw=2)

            r = [fold_name, fvector_name, clf_name, str(mean_auc)]
            print "\t".join(r)
            writer.writerow(r)
            i += 1


plt.plot([0, 1], [0, 1], '-', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Truly Native competition')
plt.legend(loc="lower right")
plt.show()


fout.close()


                




