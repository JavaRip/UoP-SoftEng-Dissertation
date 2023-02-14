import math
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
from model_utils.utils import cat_int_enc, gen_labels, impute_lower_and_median, conv_cat_str, conv_cat_num, append_test_train, split_test_train, enumerate_stratas, get_test_mlu, stratify
from model_utils.evaluator import evaluate
from model_utils.model5_agg_to_csv import label_agg_data, agg_data_to_df

def ohe_col(df, cols):
    return pd.get_dummies(data=df, columns=cols)

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()

  m5_df = agg_data_to_df('./models/model5/model/aggregate-data/')
  train = label_agg_data(m5_df, train)

  impute_lower_and_median(train)

  stratify(test)

  test['l'] = None
  test['m'] = None
  test['u'] = None

  test = get_test_mlu(train, test, 'Mouza')
  test = get_test_mlu(train, test, 'Union')
  test = get_test_mlu(train, test, 'Upazila')
  test = get_test_mlu(train, test, 'District')
  test = get_test_mlu(train, test, 'Division')
  impute_lower_and_median(test)

  test['Prediction'] = None

  for div in train_df['Division'].unique():
    tr_div = train[train['Division'] == div]
    te_div = test[test['Division'] == div]

    tt_df = append_test_train(te_div, tr_div)

    conv_cat_num(tt_df, 'Label')
    tt_df = ohe_col(tt_df, ['Mouza'])

    tt_df = tt_df.drop(
      columns=[
        'Division',
        'District',
        'Union',
        'Upazila',
      ]
    )

    cat_int_enc(tt_df)
    tt_df = pd.DataFrame(MinMaxScaler().fit_transform(tt_df), columns=tt_df.columns)

    te_div, tr_div = split_test_train(tt_df)

    train_X = tr_div.drop(['Arsenic', 'Label'], axis='columns')
    train_y = tr_div['Label']
    test_X = te_div.drop(['Arsenic', 'Label'], axis='columns')

    num_feat = len(test_X.columns)

    clf = MLPClassifier(
      solver='adam',
      alpha=0.0001,
      hidden_layer_sizes=(math.trunc(num_feat / 2), math.trunc(num_feat / 4), math.trunc(num_feat / 8)),
      learning_rate='adaptive',
      random_state=99,
      max_iter=1
    )

    clf.fit(train_X, train_y)

    test.loc[test['Division'] == div, ['Prediction']] = clf.predict(test_X)

  conv_cat_str(test, 'Prediction')
  return test['Prediction'].values

def main():
  train_src = './well_data/train.csv'
  test_src ='./well_data/test.csv'
  test_out = f'./prediction_data/model8-{time.time() / 1000}.csv';

  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src)

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df)
  evaluate(test_df)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')

if __name__ == '__main__':
  main()