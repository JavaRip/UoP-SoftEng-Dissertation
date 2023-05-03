import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, impute_lower_and_median, conv_cat_str, conv_cat_num, append_test_train, split_test_train, get_test_mlu, stratify, load_k_train
from model_utils.evaluator import gen_eval, print_eval 
from model_utils.model5_agg_to_csv import label_agg_data, agg_data_to_df

def get_name():
  return 'm8'

def ohe_col(df, cols):
    return pd.get_dummies(data=df, columns=cols)

def gen_predictions(train_df, test_df, k_fold):
  train = train_df.copy()
  test = test_df.copy()

  m5_df = agg_data_to_df(f'./models/model5/model/k{k_fold}/aggregate-data/')
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

    # if test df has no entries in this div skip
    if len(te_div) == 0:
      continue

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
      max_iter=100,
      verbose=True,
    )

    clf.fit(train_X, train_y)

    test.loc[test['Division'] == div, ['Prediction']] = clf.predict(test_X)

  conv_cat_str(test, 'Prediction')
  return test['Prediction'].values

def main(
  test_src='./well_data/k1.csv',
  k_fold=1,
):

  train_df = load_k_train(k_fold)
  test_df = pd.read_csv(test_src) 

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df, k_fold)

  return test_df

if __name__ == '__main__':
  test_out=f'./prediction_data/model8-{time.time() / 1000}.csv',
  test_df = main()

  eval = gen_eval(test_df)
  print_eval(eval)

  print(f'written to {test_out}')