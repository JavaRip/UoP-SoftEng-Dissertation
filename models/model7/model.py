import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()

  cat_int_enc(train)
  cat_int_enc(test)

  train['l'].fillna((train['l'].mode()), inplace=True)
  train['u'].fillna((train['u'].mode()), inplace=True)
  test['l'].fillna((test['l'].mode()), inplace=True)
  test['u'].fillna((test['u'].mode()), inplace=True)

  train['Label'] = gen_labels(train)
  test['Label'] = gen_labels(test)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  test_y = test['Label']

  rf_model = RandomForestClassifier(random_state=99)
  rf_model.fit(train_X, train_y)

  return rf_model.predict(test_X)

if __name__ == '__main__':
  train_src = './models/model7/train.csv'
  test_src ='./models/model7/test.csv'
  test_out = f'./prediction_data/model7-{time.time() / 1000}.csv';

  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src)

  test_df['predictions'] = gen_predictions(train_df, test_df)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
