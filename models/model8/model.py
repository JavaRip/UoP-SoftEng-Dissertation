import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, impute_lower_and_median, conv_cat_str, conv_cat_num, append_test_train, split_test_train, enumerate_stratas
from model_utils.evaluator import evaluate

def ohe_col(df, cols):
    return pd.get_dummies(data=df, columns=cols)

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()
  test['Prediction'] = None

  print('__________________________')

  for div in train_df['Division'].unique():
    print(div)
    print('##############')

    tr_div = train[train['Division'] == div]
    te_div = test[test['Division'] == div]

    tt_df = append_test_train(
      te_div,
      tr_div,
    )

    impute_lower_and_median(tt_df)
    enumerate_stratas(tt_df)

    conv_cat_num(tt_df, 'Label')
    tt_df = ohe_col(tt_df, ['Union'])

    tt_df = tt_df.drop(
      columns=[
        'Division',
        'District',
        'Upazila',
        'Mouza',
      ]
    )

    cat_int_enc(tt_df)
    tt_df = pd.DataFrame(MinMaxScaler().fit_transform(tt_df), columns=tt_df.columns)

    te_div, tr_div = split_test_train(tt_df)

    train_X = tr_div.drop(['Arsenic', 'Label'], axis='columns')
    train_y = tr_div['Label']
    test_X = te_div.drop(['Arsenic', 'Label'], axis='columns')

    clf = MLPClassifier(
      solver='adam',
      alpha=0.0001,
      hidden_layer_sizes=(100, 5),
      learning_rate='adaptive',
      random_state=99,
      verbose=True,
      max_iter=5
    )

    clf.fit(train_X, train_y)

    test_X.info()
    
    test.loc[test['Division'] == div, ['Prediction']] = clf.predict(test_X)
    test.info()
    print(test_df.head())

  print('###########################')
  conv_cat_str(test, 'Prediction')
  test.info()
  return test['Prediction']

if __name__ == '__main__':
  train_src = './models/model8/train.csv'
  test_src ='./models/model8/test.csv'
  test_out = f'./prediction_data/model8-{time.time() / 1000}.csv';

  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src)

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df)
  print('-----------------------------')
  print('-----------------------------')
  print('-----------------------------')
  evaluate(test_df)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
