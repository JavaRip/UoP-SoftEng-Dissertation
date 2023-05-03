import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import gen_labels, gen_centroids, split_test_train, append_test_train, conv_cat_num, conv_cat_str, load_k_train
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm9'

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

  test, train = split_test_train(tt_df)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')

  clf = MLPClassifier(
    solver='adam',
    alpha=0.0001,
    hidden_layer_sizes=(50, 2),
    learning_rate='adaptive',
    random_state=99,
    max_iter=100,
    verbose=True
  )

  clf.fit(train_X, train_y)
  test_X['Prediction'] = clf.predict(test_X)
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
  test_out=f'./prediction_data/model9-{time.time() / 1000}.csv',
  test_df = main()

  eval = gen_eval(test_df)
  print_eval(eval)

  print(f'written to {test_out}')
