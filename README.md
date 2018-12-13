## Project1
## Data
[http://archive.ics.uci.edu/ml/machine-learning-databases/adult/](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/)
## model
- svm
- adaboost
- xgboost
- LinearRegression
- LogitRegression
- GBDT
- lightgbm
- knn
- nn
- randomforest
- DecesionTree
- naive bayes
## Run
```shell
python classifier.py \
    --model_name {'model name'} \
    --train_path {'The path of train data'} \
    --test_path {'The path of test data'}
```
- Default `train_path`: `'./data/adult.data'`
- Default `test_path`:`'./data/adult.test'`