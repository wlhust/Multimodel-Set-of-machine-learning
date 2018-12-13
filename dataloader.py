# -*- coding: utf-8 -*- 
# author: wangli

import pandas as pd
import numpy as np
from tqdm import tqdm

class dataloader():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.col_labels = ['age','workclass','fnlwgt','education','education_num','marital_status',
                    'occupation','relationship','race','sex','capital_gain','capital_loss',
                    'hours_per_week','native_country','income']
        self.remove_cols = ['race', 'capital_loss']

    def process_test(self):
        # 由于下载的adult.test的第一行有 '|1x3 Cross validator\n'，故删掉此行
        with open(self.test_path, 'r') as f:
            contents = f.readlines()
        with open('./adult_new.test', 'w') as f:
            for line in contents[1:]:
                f.writelines(line)
        self.test_path = './adult_new.test'

    def gen_dataset(self):
        train_dataset = pd.read_csv(self.train_path, header=None)
        test_dataset = pd.read_csv(self.test_path, header=None)
        train_dataset.columns = self.col_labels
        test_dataset.columns = self.col_labels
        workclass = list(set(train_dataset['workclass']))
        marital_status = list(set(train_dataset['marital_status']))
        sex = list(set(train_dataset['sex']))
        set_capital_gain = set(train_dataset['capital_gain'])
        set_capital_gain.update(set(test_dataset['capital_gain']))
        capital_gain = list(set_capital_gain)
        education = list(set(train_dataset['education']))
        occupation = list(set(train_dataset['occupation']))
        relationship = list(set(train_dataset['relationship']))
        native_country = list(set(train_dataset['native_country']))

        def preprocess_row(row):
            if "<=50" in row["income"]:
                row["income"] = 0
            else:
                row["income"] = 1
            row["workclass"] = workclass.index(row["workclass"])
            row["marital_status"] = marital_status.index(row["marital_status"])
        #     row["race"] = race.index(row["race"])
            row["sex"] = sex.index(row["sex"])
            row["capital_gain"] = capital_gain.index(row["capital_gain"])
            row["education"] = education.index(row["education"])
            row["occupation"] = occupation.index(row["occupation"])
            row["relationship"] = relationship.index(row["relationship"])
            row["native_country"] = native_country.index(row["native_country"])
            return row  

        train_dataset = train_dataset.filter(items=set(train_dataset.columns) - {"race", "capital_loss"})
        test_dataset = test_dataset.filter(items=set(test_dataset.columns) - {"race", "capital_loss"})
        for col in self.remove_cols:
            self.col_labels.remove(col)
        # print(train_dataset)
        print('==> decoding train dataset')
        train_dataset = train_dataset.apply(preprocess_row, axis=1)
        print('==> decoding test dataset')
        test_dataset = test_dataset.apply(preprocess_row, axis=1)

        train_data = train_dataset.filter(items=set(train_dataset.columns) - {"income"})
        test_data = test_dataset.filter(items=set(test_dataset.columns) - {"income"})
        train_y = train_dataset['income']
        test_y = test_dataset['income']
        return train_data, train_y, test_data, test_y