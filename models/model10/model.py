import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import gen_labels, gen_centroids, split_test_train, append_test_train, conv_cat_num, conv_cat_str, load_k_train, stratify, enumerate_stratas
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm10'

def gen_predictions(train_df, test_df, gdf):
  train = train_df.copy()
  test = test_df.copy()

  tt_df = append_test_train(test, train)

  tt_df['lon'], tt_df['lat'] = gen_centroids(tt_df, gdf)

  tt_df.drop(
    columns=['Division', 'District', 'Upazila', 'Union', 'Mouza'], 
    inplace=True,
  )

  conv_cat_num(tt_df, 'Label')
  tt_df = pd.DataFrame(MinMaxScaler().fit_transform(tt_df), columns=tt_df.columns)
  stratify(tt_df)
  enumerate_stratas(tt_df)

  test, train = split_test_train(tt_df)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')

  knn = KNeighborsClassifier(n_neighbors=50)

  knn.fit(train_X, train_y)
  test_X['Prediction'] = knn.predict(test_X)
  conv_cat_str(test_X, 'Prediction')
  test_X.reset_index(inplace=True)

  return test_X['Prediction']

def main(
  test_src='./well_data/k1.csv',
  k_fold=1,
):
  gdf = gpd.read_file('./geodata/mou/mou-c005-s010-vw-pr.geojson')
  train_df = load_k_train(k_fold)
  test_df = pd.read_csv(test_src) 

  train_df['Label'] = gen_labels(train_df)
  test_df['Label'] = gen_labels(test_df)

  test_df['Prediction'] = gen_predictions(train_df, test_df, gdf)
  return test_df

if __name__ == '__main__':
  test_df = main()

  eval = gen_eval(test_df)
  print_eval(eval)
