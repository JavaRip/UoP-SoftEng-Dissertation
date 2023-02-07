import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, conv_cat_num, conv_cat_str, stratify, enumerate_stratas
from model_utils.evaluator import evaluate

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()

  mlu_df = train.drop(
    columns=[
      'Division', 
      'District', 
      'Upazila', 
      'Union', 
      'Depth', 
      'Arsenic', 
      'Label', 
      'Strata'
    ]).drop_duplicates(
      subset='Mouza',
    )

  test = test.merge(
    mlu_df,
    on=['Mouza'],
    how='left',
  )

  test['m'].fillna(train['m'].mean(), inplace=True)
  test['l'].fillna(train['l'].mean(), inplace=True)
  test['u'].fillna(train['u'].mean(), inplace=True)

  stratify(train)
  enumerate_stratas(train)
  stratify(test)
  enumerate_stratas(test)

  conv_cat_num(train, 'Label')
  conv_cat_num(test, 'Label')

  cat_int_enc(train)
  cat_int_enc(test)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  
  train_X = train.reindex(sorted(train_X.columns), axis=1)
  test_X = test.reindex(sorted(test_X.columns), axis=1)

  rf_model = RandomForestClassifier(random_state=99)
  rf_model.fit(train_X, train_y)

  test_X['Prediction'] = rf_model.predict(test_X)
  conv_cat_str(test_X, 'Prediction')

  return test_X['Prediction']

if __name__ == '__main__':
  train_src = './models/model7/train.csv'
  test_src ='./well_data/test.csv'
  test_out = f'./prediction_data/model7-{time.time() / 1000}.csv';

  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src)

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df)

  evaluate(test_df)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
