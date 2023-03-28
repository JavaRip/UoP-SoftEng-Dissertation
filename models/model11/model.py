import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
import time
import sys
import os

from models.model_utils.model5_agg_to_csv import agg_data_to_df, label_agg_data

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import gen_labels, gen_centroids, split_test_train, append_test_train, conv_cat_num, conv_cat_str, load_k_train, stratify, enumerate_stratas, impute_lower_and_median, get_test_mlu, cat_int_enc
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm11'

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
  knn = KNeighborsClassifier(n_neighbors=250)

  knn.fit(train_X, train_y)
  test_X['Prediction'] = knn.predict(test_X)
  conv_cat_str(test_X, 'Prediction')
  test_X.reset_index(inplace=True)

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
  test_df = main()

  eval = gen_eval(test_df)
  print_eval(eval)
