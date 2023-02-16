import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, conv_cat_num, conv_cat_str, stratify, enumerate_stratas, get_test_mlu, impute_lower_and_median, load_k_train
from model_utils.model5_agg_to_csv import label_agg_data, agg_data_to_df
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm7'

def gen_predictions(train_df, test_df, k_fold):
  train = train_df.copy()
  test = test_df.copy()

  m5_df = agg_data_to_df(f'./models/model5/model/k{k_fold}/aggregate-data/')
  train = label_agg_data(m5_df, train)

  impute_lower_and_median(train)

  test['l'] = None
  test['m'] = None
  test['u'] = None

  stratify(test)

  test = get_test_mlu(train, test, 'Mouza')
  test = get_test_mlu(train, test, 'Union')
  test = get_test_mlu(train, test, 'Upazila')
  test = get_test_mlu(train, test, 'District')
  test = get_test_mlu(train, test, 'Division')
  impute_lower_and_median(test)

  enumerate_stratas(test)
  enumerate_stratas(train)

  conv_cat_num(train, 'Label')
  conv_cat_num(test, 'Label')

  cat_int_enc(train)
  cat_int_enc(test)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']
  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  
  train_X = train.reindex(sorted(train_X.columns), axis=1)
  test_X = test.reindex(sorted(test_X.columns), axis=1)

  train_X.reset_index(inplace=True)
  test_X.reset_index(inplace=True)

  train_X = train_X.drop(columns=['index'])
  test_X = test_X.drop(columns=['index'])

  rf_model = RandomForestClassifier(random_state=99)
  rf_model.fit(train_X, train_y)

  test_X['Prediction'] = rf_model.predict(test_X)
  conv_cat_str(test_X, 'Prediction')

  return test_X['Prediction']

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
  test_out=f'./prediction_data/model7-{time.time() / 1000}.csv',
  test_df = main()

  eval = gen_eval(test_df)
  print_eval(eval)

  print(f'written to {test_out}')