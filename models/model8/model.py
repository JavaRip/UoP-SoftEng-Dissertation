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
  for col in cols:
    enc = OneHotEncoder(handle_unknown='ignore')

    col_enc_df = pd.DataFrame(enc.fit_transform(df[[col]]).toarray())
    col_enc_df.columns = enc.get_feature_names_out([col])
    col_enc_df.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True, drop=True)
    df = pd.concat([df, col_enc_df], axis=1)
  return df

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()
  train.info()
  test.info()
  print('=========================')

  tt_df = append_test_train(test, train)

  impute_lower_and_median(tt_df)
  enumerate_stratas(tt_df)

  conv_cat_num(tt_df, 'Label')
  tt_df = ohe_col(tt_df, ['Union'])

  tt_df = tt_df.drop(
    columns=[
      'Division',
      'District',
      'Upazila',
      'Union',
      'Mouza',
    ]
  )

  cat_int_enc(tt_df)
  tt_df = pd.DataFrame(MinMaxScaler().fit_transform(tt_df), columns=tt_df.columns)

  test, train = split_test_train(tt_df)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']
  test_X = test.drop(['Arsenic', 'Label'], axis='columns')

  clf = MLPClassifier(
    solver='adam',
    alpha=0.0001,
    hidden_layer_sizes=(100, 5),
    learning_rate='adaptive',
    random_state=99,
    verbose=True
  )

  clf.fit(train_X, train_y)
  test['Prediction'] = clf.predict(test_X)
  print('~~~~~~~~~~~~~~~~~~')
  test['Prediction'].info()
  print(test['Prediction'].unique())
  conv_cat_str(test, 'Prediction')
  print(test['Prediction'].unique())
  print(test['Prediction'].info())

  return test['Prediction'].squeeze()

if __name__ == '__main__':
  train_src = './models/model8/train.csv'
  test_src ='./models/model8/test.csv'
  test_out = f'./prediction_data/model8-{time.time() / 1000}.csv';

  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src)

  train_df = train_df[train_df['Division'] == 'Chittagong']
  test_df = test_df[test_df['Division'] == 'Chittagong']
  
  train_df.reset_index(inplace=True)
  test_df.reset_index(inplace=True)

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df)

  evaluate(test_df)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
