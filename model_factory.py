# -*- coding: utf-8 -*-
# author: wangli

from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

class classifier_model():
    def __init__(self, model_name):
        self.model_name = model_name
        self.clf_DT = tree.DecisionTreeClassifier()
        self.clf_svm = LinearSVC()
        self.clf_adaboost = AdaBoostClassifier()
        self.clf_xgboost = XGBClassifier(learning_rate=0.1)
        self.clf_LinearR = LinearRegression()
        self.clf_LR = LinearRegression()
        self.clf_gbdt = GradientBoostingClassifier()
        self.gbm = lgb.LGBMClassifier()
        self.clf_knn = KNeighborsClassifier()
        self.clf_perception = Perceptron()
        self.clf_RF = RandomForestClassifier()
        self.naive_bayes = GaussianNB()
    
    def get_classifier_fn(self):
        model_dict = {
            'svm':          self.clf_svm,
            'adaboost':     self.clf_adaboost,
            'xgboost':      self.clf_xgboost,
            'linearR':      self.clf_LinearR,
            'LR':           self.clf_LR,
            'gbdt':         self.clf_gbdt,
            'lgbm':         self.gbm,
            'knn':          self.clf_knn,
            'nn':           self.clf_perception,
            'randomforest': self.clf_RF,
            'decesiontree': self.clf_DT,
            'naive_bayes' : self.naive_bayes            
        }
        return model_dict[self.model_name]