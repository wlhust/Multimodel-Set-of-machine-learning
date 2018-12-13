# -*- coding: utf-8 -*-
# author: wangli

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from dataloader import *
from model_factory import *
import argparse
import warnings
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")

pareser = argparse.ArgumentParser()
pareser.add_argument("--model_name", type=str, help='The classifier name')
pareser.add_argument('--train_path', type=str, default='./data/adult.data', help='The path of train data')
pareser.add_argument('--test_path', type=str, default='./data/adult.test', help='The path of test data')
args = pareser.parse_args()

if __name__ == "__main__":
    train_path = args.train_path
    test_path = args.test_path
    dataloader = dataloader(train_path, test_path)
    print('==> processing test')
    dataloader.process_test()
    print("==> generating train and test data")
    train_data, train_y, test_data, test_y = dataloader.gen_dataset()

    def score_LR(model, test_data, test_y):
        pred = list(map(lambda x: 0 if x < 0.5 else 1, list(model.predict(test_data))))
        label = test_y
        return list((np.array(pred) - np.array(label))).count(0) /  len(pred)

    classifier_factory = classifier_model(args.model_name)
    print('training ' + args.model_name)
    clf = classifier_factory.get_classifier_fn()
    clf.fit(train_data.as_matrix(), train_y.as_matrix())
    labels = clf.predict(test_data)
    print('==> predict......')
    use_score_LR = ['linearR', 'LR']
    if args.model_name in use_score_LR:
        score = score_LR(clf, test_data, test_y)
        print('==> Score:{}'.format(score))
    else:
        score = clf.score(test_data, test_y.as_matrix())
        corss_score = cross_val_score(clf, test_data, test_y, cv=3)
        print('==> Score: {}'.format(score))
        print('==> Cross Score: {}'.format(corss_score))
    print('==> Done!')