import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, conv_cat_num, conv_cat_str
from model_utils.evaluator import evaluate

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()

  conv_cat_num(train, 'Label')
  conv_cat_num(test, 'Label')

  cat_int_enc(train)
  cat_int_enc(test)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  test_y = test['Label']

  rf_model = RandomForestClassifier(random_state=99)
  rf_model.fit(train_X, train_y)

  test_X['Prediction'] = rf_model.predict(test_X)
  conv_cat_str(test_X, 'Prediction') 

  return test_X['Prediction']

if __name__ == '__main__':
  train_src = './well_data/train.csv'
  test_src ='./well_data/test.csv'
  test_out = f'./prediction_data/model6-{time.time() / 1000}.csv';

  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src) 

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df)

  evaluate(test_df)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
