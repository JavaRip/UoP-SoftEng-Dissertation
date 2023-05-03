
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os
import geopandas as gpd
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler 

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, conv_cat_num, conv_cat_str, load_k_train, stratify, append_test_train, gen_centroids, split_test_train
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm13'

def gen_predictions(train_df, test_df):
  train = train_df.copy()
  test = test_df.copy()
  gdf = gpd.read_file('./geodata/mou/mou-c005-s010-vw-pr.geojson')

  tt_df = append_test_train(test, train)
  tt_df['lon'], tt_df['lat'] = gen_centroids(tt_df, gdf)
  tt_df = tt_df.drop(
    columns=[
      'Division',
      'District',
      'Union',
      'Upazila',
      'Mouza'
    ]
  )
  test, train = split_test_train(tt_df)

  conv_cat_num(train, 'Label')
  conv_cat_num(test, 'Label')

  cat_int_enc(train)
  cat_int_enc(test)

  stratify(test)
  stratify(train)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')

  for strata in train['Strata'].unique():
    tr_div = train[train['Strata'] == strata]
    te_div = test[test['Strata'] == strata]

    tt_df = append_test_train(te_div, tr_div)
    cat_int_enc(tt_df)
    tt_df = pd.DataFrame(MinMaxScaler().fit_transform(tt_df), columns=tt_df.columns)
    te_div, tr_div = split_test_train(tt_df)

    train_X = tr_div.drop(['Arsenic', 'Label', 'Prediction', 'Strata', 'Depth'], axis='columns', errors='ignore')
    train_y = tr_div['Label']
    test_X = te_div.drop(['Arsenic', 'Label', 'Strata', 'Depth'], axis='columns')

    knn = KNeighborsClassifier(
      n_neighbors=5000,
      weights='uniform', 
      n_jobs=-1, 
    )
    knn.fit(train_X, train_y)
    predictions = knn.predict(test_X.loc[:, test_X.columns != 'Prediction'])

    test.loc[test['Strata'] == strata, ['Prediction']] = predictions

  conv_cat_str(test, 'Prediction')
  return test['Prediction']

def main(
  test_src='./well_data/k1.csv',
  k_fold=1,
):

  train_df = load_k_train(k_fold)
  test_df = pd.read_csv(test_src) 

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df)

  return test_df

if __name__ == '__main__':
  test_df = main()

  eval = gen_eval(test_df)
  print_eval(eval)