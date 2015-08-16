# -*- coding: utf-8 -*-

"""
file containing all useful utils for dataset setup preparation
to be used in the lexiconevaluation script file
dataset setup preparation doesn't handle text preprocessing but :
- it handles reading of files names from training file
- reading all resulting files and extracting getting text inside it.
- handles percentages of training/testing and using cross validation or not.
text preprocessing in handeled in the Document.preprocess static function
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold, train_test_split


def read_data_files(filenames, datapath, ids=None):
    """
    :param filename: list or iterator of file names
    :param datapath: datapath to concatinate before each filename
    :return: iterator of text containing all text in all files in order
    """
    if ids is None:
        ids = range(0, len(filenames))

    for i in [filenames[k] for k in ids]:
        yield str(open(datapath+i, 'r').read())


def create_cv_dataset(filename, datapath, n_folds=5):
    """
    :param filename: path of the training file that has list of all training examples
    :param datapath: data path that has all training files
    :param cv: either using cross validation or not
    :param n_folds: number of folds in case of cross validation
    :return: array of tuples in form of (X_train, y_train, X_test, y_test) where
    X is the document text and y is the polarity
    """

    df = pd.read_csv(filename, encoding="utf-8")

    y = np.array(df.sponsored)
    kf = StratifiedKFold(y, n_folds=n_folds, shuffle=False, random_state=2)

    kfolds_data=[]
    for trainids, testids in kf:

        kfolds_data.append((
            read_data_files(list(df.file), datapath, trainids),
            y[trainids],
            read_data_files(list(df.file), datapath, testids),
            y[testids])
        )

    return kfolds_data


def create_train_test_dataset(filename, datapath, percentage=0.2):
    """
    dataset takes percentage of the data and use it to create test dataset
     test dataset can be used as a validation dataset also

    :param filename: path of the training file that has list of all training examples
    :param datapath: data path that has all training files
    :param percentage: test set percentage from the whole dataset
    """

    df = pd.read_csv(filename, encoding="utf-8")

    y = np.array(df.sponsored)
    X_train, X_test, y_train, y_test = train_test_split(list(df.file), y,
                                                        test_size=percentage, random_state=2)

    return (
        read_data_files(X_train, datapath),
        read_data_files(X_test, datapath),
        y_train,
        y_test
    )


def create_unlabeled_dataset(filename, datapath):
    """
    return an iterators for all file text in the file name sent  under the file path
    basically for test datasets that is unlabeled to upload on kaggle

    :param filename: path of the training file that has list of all training examples
    :param datapath: data path that has all training files
    """

    df = pd.read_csv(filename, encoding="utf-8")
    return read_data_files(list(df.file), datapath)


